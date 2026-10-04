"""Semantle Word Hunter: plays Hebrew Semantle (https://semantle.ishefi.com) for you, driven by an AI model.

Each round the model sees every guess so far (sorted by similarity) plus a digest of what earlier games
taught it, and chooses and proposes a few new Hebrew words. The script submits them to the site's
/api/distance endpoint and loops until the secret word is found (distance == 1000).

Everything learned is stored in knowledge.json: for every game its counts, the secret word, a summary and short
lessons the model writes when the game ends (the raw guesses are not kept), and words the game rejected. The next run feeds a digest of that file
back into the prompt, so the solver improves over time. Progress is saved after every guess, so an
interrupted game is saved, and running again always starts a new one.

Run it in the terminal (python solver.py) or with a live web UI (python solver.py --ui).
"""

import argparse
import copy
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import requests

import providers

try:  # only needed for Claude through the API; Claude Code, OpenAI, Google and Ollama work without it
    import anthropic
except ImportError:
    anthropic = None

SITE = "https://semantle.ishefi.com"
HEADERS = {"X-SH-Version": "2023-09-10"}
KNOWLEDGE_PATH = Path(__file__).with_name("knowledge.json")
API_ERRORS = (anthropic.APIStatusError,) if anthropic else ()
STALL_LIMIT = 5  # consecutive Claude rounds with zero accepted words before giving up
# Effort experiments: now and then a round runs above the effort the supervisor chose, so the record shows what
# effort is really worth. Random, so the comparison is fair; capped, so the cost stays small.
EXPERIMENT_CHANCE = 0.5   # chance that an eligible round becomes an experiment
EXPERIMENT_SAMPLES = 8    # a level with this many recorded rounds needs no more experiments
MAX_EXPERIMENTS = 2       # per game
EXPERIMENT_CEILING = "high"
rng = random.Random()
MIN_GUESSES, MAX_GUESSES = 1, 10  # the model chooses the round size inside these bounds
EFFORTS = ["low", "medium", "high", "xhigh", "max"]
HAIKU_THINKING_BUDGET = 8000  # Haiku 4.5 has no adaptive thinking; the other models think adaptively

# --model takes an alias (sonnet, opus, haiku, fable) or any full model ID, so new models need no code change.
# Claude Code resolves an alias to its newest model itself. For the API backend, resolve_model() looks the
# alias up in the Models API; this table is only the last-resort fallback when that is unreachable.
MODELS = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "haiku": "claude-haiku-4-5",
    "fable": "claude-fable-5-1",
}

SYSTEM = """You are an expert player of Hebrew Semantle (semantle.ishefi.com). A secret Hebrew word is chosen
each day. You never see it; you only see how close each guess is to it in a word-embedding space, and you
must find it in as few guesses as possible.

# The feedback you get
- similarity: cosine similarity x100. In practice unrelated common words score roughly 10-30, a word in
  the right general field 35-48, and a close neighbour 50-75. Scores vary a little from day to day.
- distance (rank): only the 1000 words nearest to the secret get one, from 1 (far end) to 999 (nearest
  neighbour). The user message states today's real reference scores (nearest word, 10th nearest, 1000th
  nearest): a guess needs roughly the 1000th-nearest score to get a rank at all. 1000 means the guess IS
  the secret. -1 means "not in the top 1000".
- So until a rank appears, the similarity number is your only compass. Once ranks appear, they are far more
  informative than raw similarity: build on the highest-ranked words.

# What the score actually measures
Semantic relatedness learned from text usage, not spelling, not letters, not rhyme. Words in the same
topic, or that fill the same role in sentences, score high: "doctor" ~ "hospital" ~ "nurse". Opposites
often score high too (love/hate, day/night) because they appear in the same contexts. The secret is
almost always a common, everyday word (noun, verb or adjective) in its base form.

# How to search (work in phases, and say which phase you are in)
1. EXPLORE (no score above ~40 yet): probe widely different fields, one clear representative word each,
   e.g. a body part, an animal, a place, a tool, a food, a family role, an emotion, an abstract idea, a
   verb of motion, a time word, a nature word, a profession. Do not spend two guesses on the same field.
2. LOCATE (best score 40-48): the field is known, the exact area is not. Probe different sub-areas of
   the best field. Compare: if A scores higher than B, the secret is on A's side. Combine the top
   scorers: what concept do they share that the weaker words lack? Guess that concept directly.
3. CONVERGE (a rank appears): stay close to the highest-ranked words. Guess their synonyms, hypernyms
   (the broader category), hyponyms (specific kinds), parts/wholes, typical actions, and the single basic
   word that all the best-ranked words point to. Often the answer is simpler and more basic than the
   words that found it.
- Treat low scorers as information too: a very low score rules out a whole field; do not return to it.
- Each batch should mix: mostly the best next step, plus one deliberately different probe whenever the
  search has stalled (several guesses with no improvement).
- Never fixate on spelling variants or inflections of one word; the embedding groups them.

# Hebrew rules for each guess
- Exactly one word: no spaces, no niqqud, no punctuation, no Latin letters.
- Base form: singular masculine noun, masculine singular adjective, past 3rd-person masculine verb or the
  infinitive. No attached prefixes (ה, ב, ל, ו, מ, כ, ש) unless the word is normally written with them.
- The game's vocabulary is limited. If a word is rejected, the next try should be a plainer, more common
  word with the same meaning, or a different spelling (full/defective), not the same word again.
- Never repeat any word from the history or the rejected list.

# Using what you remember
The user message may start with "Knowledge from previous games": past secret words, a summary of each recent
game, how each scout's words fared, and lessons. Use it:
- Read the summaries: they show how earlier searches moved from the field to the answer and what wasted guesses.
  Imitate what worked, avoid what wasted guesses.
- Today's secret is a different word from every past secret. Past secrets are only evidence about how
  the embedding behaves, never candidates.
- Lessons are guidance from small samples: follow them, but let this game's actual scores override them.

# How many guesses to submit each round
You choose the number of guesses in every round, from 1 to 10, and the game is scored by the total number of
guesses, so spend them deliberately. A round costs a model call and some time; a guess costs one point.
- Mapping the territory (no score near the 1000th-nearest reference score yet): a wide batch of 6 to 10
  probes from different fields finds the field fastest, and the probes do not depend on each other.
- Once a field or a ranked word is known: small batches of 1 to 3, because every result should steer the
  next guess. Do not add filler words to reach a number.
- When you are sure of a single best candidate, guess just that.
State the number you chose and why in your reasoning.

# Sub-agents
Each round, sub-agents (scouts) may have explored in parallel before you. When the user message contains a
"Sub-agent proposals" section, USE it every round: it is cheap parallel exploration you are expected to
exploit. The scouts are the field-scout (fields not probed yet), the neighbour-scout (close relatives of the
best words) and the triangulator (what the best words share that the weak ones lack, including the outcome,
the opposite and the cause). Take the strongest proposals, give extra weight to words several scouts
propose, and add your own where you see better. You are responsible for the final batch, since scouts can be
wrong. When your own recent guesses plateau, lean on the scout that explores a different axis than you did.

# Output
Reply only with the requested JSON: a short "reasoning" (2-4 sentences: current phase, what the scores
suggest, why these words) and the "guesses" list with the words you chose to submit this round."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "guesses": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "guesses"],
    "additionalProperties": False,
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "lessons": {"type": "array", "items": {"type": "string"}}},
    "required": ["summary", "lessons"],
    "additionalProperties": False,
}

LESSON_SCHEMA = {
    "type": "object",
    "properties": {"lessons": {"type": "array", "items": {"type": "string"}}},
    "required": ["lessons"],
    "additionalProperties": False,
}


# ---------- knowledge file ----------

def load_knowledge(path: Path) -> dict:
    know = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"version": 1, "rejected": [], "games": []}
    know.setdefault("lessons", [])             # lessons kept independently of any one game's record
    know.setdefault("supervisor_lessons", [])  # what the supervisor learned about its own effort decisions
    return know


def save_knowledge(path: Path, know: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # a custom --knowledge path may point into a new folder
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(know, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except PermissionError:  # Windows: another process (the UI reading the file, a scanner) holds it briefly
            time.sleep(0.05 * (attempt + 1))
    path.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")  # last resort: write in place
    tmp.unlink(missing_ok=True)


def format_path(guesses: list[dict], head: int = 8, tail: int = 12) -> str:
    """Compact one-guess-per-line trail; long games keep the opening and the final approach."""
    rows = [f"{i}. {x['guess']}  {x['similarity']:.1f}  {x['distance'] if x['distance'] and x['distance'] > 0 else '-'}"
            for i, x in enumerate(guesses, 1)]
    if len(rows) > head + tail:
        rows = rows[:head] + [f"   ... ({len(rows) - head - tail} guesses omitted) ..."] + rows[-tail:]
    return "\n".join(rows)


def guess_count(g: dict) -> int:
    return g.get("guess_count", len(g.get("guesses", [])))


def compact_game(g: dict) -> dict:
    """A game as it is stored: counts and a summary, never the raw guesses."""
    out = {k: v for k, v in g.items() if k != "guesses"}
    guesses = g.get("guesses")
    if guesses is not None:
        out["guess_count"] = len(guesses)
        out["best_similarity"] = max((x["similarity"] for x in guesses), default=None)
        out["best_rank"] = max((x["distance"] for x in guesses if x["distance"] and x["distance"] > 0), default=None)
    return out


def build_memory(know: dict, current: dict) -> str:
    """Digest of earlier games for the prompt: past secrets, a summary of each recent game, scouts, lessons."""
    past = [g for g in know["games"] if g is not current]
    if not past and not know.get("lessons"):
        return ""
    parts = []

    # The same puzzle can be played more than once; keep the most efficient solve per secret word.
    best: dict[str, dict] = {}
    for g in past:
        if g.get("solved") and g.get("secret") and (g["secret"] not in best
                                                    or guess_count(g) < guess_count(best[g["secret"]])):
            best[g["secret"]] = g
    solved = [g for g in past if g.get("solved") and g.get("secret") and best[g["secret"]] is g]
    if solved:
        parts.append("Past secret words (never the answer again, probably): "
                     + ", ".join(f"{g['secret']} ({guess_count(g)} guesses)" for g in solved[-30:]))

    summaries = [g for g in past if g.get("summary")][-8:]
    if summaries:
        parts.append("What happened in earlier games (the raw guesses are not kept; these summaries are):\n"
                     + "\n".join(f"- {'solved ' + g['secret'] if g.get('solved') else 'not solved'} in "
                                 f"{guess_count(g)} guesses: {g['summary']}" for g in summaries))

    # What each sub-agent's words were worth, from the rounds recorded in earlier games.
    track: dict[str, list] = defaultdict(lambda: [0, 0.0, 0])
    for g in past:
        for r in g.get("rounds", []):
            for role, info in (r.get("scouts") or {}).items():
                used = info.get("used")
                if isinstance(used, list):  # older records kept the words themselves
                    n, total, ranked = len(used), sum(u[1] for u in used), sum(1 for u in used if u[2] and u[2] > 0)
                else:
                    n, total, ranked = used or 0, info.get("sum_sim", 0.0), info.get("ranked", 0)
                track[role][0] += n
                track[role][1] += total
                track[role][2] += ranked
    used = [f"{role}: {n} words used, average similarity {total / n:.1f}, {ranked} with a rank"
            for role, (n, total, ranked) in track.items() if n]
    if used:
        parts.append("Sub-agent track record from earlier games (words you took from each scout): " + "; ".join(used))

    lessons = list(know.get("lessons", [])) + [l for g in past[-10:] for l in g.get("lessons", [])]
    if lessons:
        parts.append("Lessons from earlier games:\n" + "\n".join(f"- {l}" for l in lessons[-25:]))

    return "Knowledge from previous games:\n" + "\n\n".join(parts) + "\n\n"


def knowledge_stats(know: dict) -> list[dict]:
    """One row per game for the UI's statistics: how long it took and under which setup."""
    out = []
    for g in know["games"]:
        rounds = g.get("rounds", [])
        models = [r["model"] for r in rounds] or [g.get("model")]
        out.append({
            "puzzle": g.get("puzzle"), "date": g.get("date"), "solved": bool(g.get("solved")),
            "guesses": guess_count(g), "rounds": len(rounds),
            "model": max(set(models), key=models.count),  # the model that played most of the rounds
            "subagents": (g.get("config") or {}).get("subagents",
                                                     max((len(r.get("scouts") or {}) for r in rounds), default=0)),
            "supervisor": bool(g.get("supervisor_log")),
            "replay": bool(g.get("replay")),
        })
    return out


def load_races(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def append_race(path: Path, race: dict) -> None:
    races = load_races(path)
    races.append(race)
    path.write_text(json.dumps(races, ensure_ascii=False, indent=1), encoding="utf-8")


def knowledge_summary(know: dict, races: list[dict] | None = None) -> dict:
    """What the web UI shows in its memory panel."""
    games = [{"puzzle": g.get("puzzle"), "date": g.get("date"), "secret": g.get("secret"),
              "solved": g.get("solved"), "guesses": guess_count(g), "model": g.get("model")}
             for g in know["games"]]
    lessons = (know.get("lessons", []) + [l for g in know["games"][-10:] for l in g.get("lessons", [])])[-12:]
    return {"games": games[-30:], "lessons": lessons, "rejected": len(know["rejected"]),
            "stats": knowledge_stats(know), "races": (races or [])[-12:]}


# ---------- site ----------

def get_puzzle_info() -> tuple[int | None, dict | None]:
    """Today's puzzle number and the reference scores the home page publishes for it."""
    try:
        html = requests.get(SITE, timeout=15).text
    except requests.RequestException:
        return None, None
    m = re.search(r'id="puzzleNumber">(\d+)<', html)
    num = r"(\d+(?:\.\d+)?)"  # not [\d.]+: the page ends the sentence with a period ("48.95.")
    ref = (re.search(r"\(999/1000\)[^<]*<b>" + num + "</b>", html),
           re.search(r"\(990/1000\)[^\d]*" + num, html),
           re.search(r"\(1/1000\)[^\d]*" + num, html))
    thresholds = None
    if all(ref):
        thresholds = {"nearest": float(ref[0].group(1)), "tenth": float(ref[1].group(1)),
                      "thousandth": float(ref[2].group(1))}
    return (int(m.group(1)) if m else None), thresholds


def get_distance(word: str) -> dict | None:
    """Returns the site's record for a word, or None if the word is not in the vocabulary."""
    for attempt in range(3):
        try:
            r = requests.get(f"{SITE}/api/distance", params={"word": word}, headers=HEADERS, timeout=15)
        except requests.RequestException:
            time.sleep(1 + attempt)
            continue
        if r.status_code == 200:
            data = r.json()
            return data[0] if data else None
        if r.status_code == 400:
            return None
        time.sleep(1 + attempt)
    return None


# ---------- Claude ----------

def is_haiku(model: str) -> bool:
    return "haiku" in model.lower()


_models_cache: tuple[float, list[dict]] = (0.0, [])


def list_models() -> list[dict]:
    """Models your account can use, newest first, from the Anthropic Models API ([] when unreachable).
    Credentials: ANTHROPIC_API_KEY, else the `ant auth login` profile (needs the `ant` CLI on PATH or in tools/)."""
    global _models_cache
    if time.time() - _models_cache[0] < 600 and _models_cache[1]:
        return _models_cache[1]
    headers = {"anthropic-version": "2023-06-01"}
    try:
        if key := os.environ.get("ANTHROPIC_API_KEY"):
            headers["x-api-key"] = key
        else:
            ant = shutil.which("ant") or next((str(p) for p in (Path(__file__).parent / "tools").glob("ant*")
                                               if p.is_file() and p.suffix in ("", ".exe")), None)
            if not ant:
                return []
            token = subprocess.run([ant, "auth", "print-credentials", "--access-token"], capture_output=True,
                                   text=True, timeout=20).stdout.strip()
            if not token:
                return []
            headers.update({"Authorization": f"Bearer {token}", "anthropic-beta": "oauth-2025-04-20"})
        r = requests.get("https://api.anthropic.com/v1/models", params={"limit": 100}, headers=headers, timeout=20)
        if r.status_code != 200:
            return []
        models = [{"id": m["id"], "label": m.get("display_name") or m["id"]} for m in r.json().get("data", [])]
    except (requests.RequestException, subprocess.SubprocessError, OSError, ValueError, KeyError):
        return []
    _models_cache = (time.time(), models)
    return models


def model_options() -> list[dict]:
    """What the UI offers: Claude aliases (always the newest of each family), every Claude model the API
    lists, then the models of every other provider that is reachable."""
    aliases = [{"id": a, "label": f"{a.capitalize()} (newest)", "group": "Claude · newest"} for a in MODELS]
    claude = [{**m, "group": "Claude"} for m in list_models()]
    return aliases + claude + providers.list_provider_models()


def resolve_model(name: str) -> str:
    """Full model ID for the API backend. Claude Code accepts aliases itself, so the cli backend skips this."""
    if name not in MODELS:
        return name
    for m in list_models():  # newest first
        if m["id"].startswith(f"claude-{name}"):
            return m["id"]
    return MODELS[name]


def find_claude_cli(explicit: str | None) -> str:
    """Locate the Claude Code binary: --claude-bin, PATH, or the VS Code extension's bundled copy."""
    if explicit:
        return explicit
    if found := shutil.which("claude"):
        return found
    bundled = sorted(Path.home().glob(".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude*"))
    if bundled:
        return str(bundled[-1])
    raise RuntimeError("Claude Code binary not found; pass --claude-bin or use --backend api")


def call_claude_cli(args, system: str, prompt: str, schema: dict, effort: str) -> dict:
    """Runs one non-interactive Claude Code call, which draws on the logged-in subscription's usage limit.
    Tools, MCP servers, settings and slash commands are all disabled so each call stays ~3K tokens."""
    cmd = [find_claude_cli(args.claude_bin), "-p", "--model", args.model, "--output-format", "json",
           "--json-schema", json.dumps(schema), "--system-prompt", system, "--tools", "",
           "--disable-slash-commands", "--no-session-persistence", "--setting-sources", "",
           "--settings", '{"alwaysThinkingEnabled": true}',  # extended thinking always on
           "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
    env = dict(os.environ)
    if is_haiku(args.model):
        env["MAX_THINKING_TOKENS"] = str(HAIKU_THINKING_BUDGET)  # Haiku needs an explicit thinking budget
    else:
        cmd += ["--effort", effort]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                          cwd=tempfile.gettempdir(), timeout=600, env=env)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {(proc.stderr or proc.stdout)[:500]}")
    if out.get("is_error"):
        raise RuntimeError(f"claude CLI error: {out.get('result')}")
    data = out.get("structured_output") or json.loads(out["result"])
    thinking = (out.get("usage", {}).get("output_tokens_details") or {}).get("thinking_tokens")
    return {**data, "_meta": {"thinking_tokens": thinking}}


def effort_ok(model: str) -> bool:
    """Does this model have an effort level a supervisor can adjust?"""
    provider, name = providers.split_model(model)
    return providers.effort_supported(provider, name) if provider else not is_haiku(model)


def call_model(client, args, system: str, prompt: str, schema: dict, effort: str) -> dict:
    """One structured call to whatever model args.model names: Claude, or 'provider:model' for the others."""
    provider, name = providers.split_model(args.model)
    if provider:
        return providers.call(args, provider, name, system, prompt, schema, effort)
    backend = args.backend
    if backend == "auto":  # subscription through Claude Code when it is installed, otherwise an API key
        try:
            find_claude_cli(args.claude_bin)
            backend = "cli"
        except RuntimeError:
            backend = "api"
    if backend == "cli":
        return call_claude_cli(args, system, prompt, schema, effort)
    if anthropic is None:
        raise RuntimeError("Claude through the API needs the anthropic package: pip install anthropic")
    client = client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (or an `ant auth login` profile)
    output_config = {"format": {"type": "json_schema", "schema": schema}}
    if is_haiku(args.model):  # Haiku 4.5 rejects effort and needs a thinking budget
        thinking = {"type": "enabled", "budget_tokens": HAIKU_THINKING_BUDGET}
    else:
        output_config["effort"] = effort
        thinking = {"type": "adaptive"}
    response = client.messages.create(
        model=resolve_model(args.model),
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        thinking=thinking,  # extended thinking always on
        output_config=output_config,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Model refused: {response.stop_details}")
    return json.loads(next(b.text for b in response.content if b.type == "text"))


def build_prompt(memory: str, history: list[dict], rejected: list[str],
                 thresholds: dict | None, scouts: dict | None = None) -> str:
    ranked = sorted(history, key=lambda h: h["similarity"], reverse=True)
    shown = ranked[:100]
    shown += [h for h in history[-20:] if h not in shown]  # always include the latest probes
    lines = []
    for h in shown:
        rank = f"{h['distance']}/1000" if h["distance"] and h["distance"] > 0 else "-"
        lines.append(f"{h['guess']}\t{h['similarity']:.2f}\t{rank}")
    table = "\n".join(lines) if lines else "(no guesses yet)"
    omitted = len(history) - len(shown)
    note = f" (+{omitted} lower-scoring guesses omitted; do not repeat any word)" if omitted > 0 else ""
    ref = ""
    if thresholds:
        ref = (f"Today's real reference scores: nearest word {thresholds['nearest']}, 10th nearest "
               f"{thresholds['tenth']}, 1000th nearest {thresholds['thousandth']}. A guess needs about "
               f"{thresholds['thousandth']} to receive a rank.\n\n")
    proposals = ""
    if scouts:
        by_word: dict[str, list[str]] = {}
        for role, words in scouts.items():
            for w in words:
                by_word.setdefault(w, []).append(role)
        several = [w for w, roles in by_word.items() if len(roles) > 1]
        proposals = ("Sub-agent proposals (scouts explored in parallel; use them, you decide the final batch):\n"
                     + "\n".join(f"- {role}: {', '.join(words)}" for role, words in scouts.items())
                     + (f"\nProposed by more than one scout: {', '.join(several)}" if several else "") + "\n\n")
    return (
        f"{memory}{ref}"
        f"Guesses this game ({len(history)}){note}, format: word<TAB>similarity<TAB>rank:\n{table}\n\n"
        f"Rejected recently (not in the game's vocabulary): {', '.join(rejected[-50:]) or '(none)'}\n\n"
        f"{proposals}"
        f"Choose how many new Hebrew words to guess this round ({MIN_GUESSES} to {MAX_GUESSES}) and propose them. "
        "Keep reasoning short (a few sentences)."
    )


def write_lessons(client, args, game: dict, know: dict, solved: bool) -> dict:
    """Summarises the game and writes lessons. The raw guesses are not kept, so everything worth keeping goes here."""
    earlier = (list(know.get("lessons", [])) + [l for g in know["games"] if g is not game for l in g.get("lessons", [])])[-25:]
    count = len(game["guesses"])
    outcome = (f"You just solved a Hebrew Semantle puzzle in {count} guesses. The secret word was: {game['secret']}."
               if solved else f"A Hebrew Semantle game just ended without finding the word, after {count} guesses.")
    prompt = (
        f"{outcome}\n"
        f"Your guesses in order (word, similarity, rank):\n{format_path(game['guesses'], head=60, tail=60)}\n\n"
        f"Lessons already recorded from earlier games:\n"
        f"{chr(10).join('- ' + l for l in earlier) or '(none yet)'}\n\n"
        "The raw guesses of this game will NOT be kept, so write down everything worth keeping from them.\n"
        "1. summary: 4 to 7 plain sentences for a future player who has not seen this game: "
        + ("the secret word and how many guesses it took, " if solved else "that the word was not found, ")
        + "the field it belongs to, the words that ranked closest with their ranks, the turning point, what was "
        "wasted, and how the scouts and the effort mattered.\n"
        "2. lessons: 2-4 NEW one-sentence lessons for solving future puzzles faster. Do not repeat an existing lesson; "
        "if this game contradicts one, say so explicitly. One game is a small sample: phrase lessons about the "
        "search process (what to try when), not about what kind of word secrets usually are."
    )
    return call_model(client, args, SYSTEM, prompt, SUMMARY_SCHEMA, "low")


# ---------- sub-agents (scouts) ----------

SCOUT_SYSTEM = """You are a scout working for an automatic Hebrew Semantle solver. The solver is trying to find a
secret Hebrew word; every guess gets a similarity score (higher is closer in a word-embedding space) and the
1000 nearest words also get a rank. You see the guesses so far and propose new candidate words in your
assigned role. Propose single Hebrew words in their base form (no spaces, no niqqud, no prefixes), common
everyday words, never one that was already guessed. Reply with the requested JSON only."""

SCOUT_ROLES = [
    ("field-scout", "You are the FIELD scout. Propose words from semantic fields that have NOT been probed yet, "
                    "or only barely (look at the guesses: which fields are missing?). Give one clear, common "
                    "representative word per field. If the best score is already near the 1000th-nearest "
                    "reference score, probe sub-fields adjacent to the best field instead."),
    ("neighbour-scout", "You are the NEIGHBOUR scout. Propose the closest relatives of the three best-scoring "
                        "words: synonyms, broader categories, specific kinds, parts and wholes. Skip "
                        "inflections, construct forms and spelling variants; they score like the base word."),
    ("triangulator", "You are the TRIANGULATOR. Work out what the highest-scoring words have in common that the "
                     "low-scoring words lack, then propose words that embody that shared concept from a different "
                     "angle: its outcome, its cause, its opposite, its typical actor or setting. If the top words "
                     "are all near-synonyms of one action, jump to the result or to the opposite side of it."),
]

SCOUT_SCHEMA = {
    "type": "object",
    "properties": {"note": {"type": "string"}, "candidates": {"type": "array", "items": {"type": "string"}}},
    "required": ["note", "candidates"],
    "additionalProperties": False,
}


def run_scouts(client, args, history: list[dict], rejected: list[str], thresholds: dict | None, emit,
               count: int, model: str) -> dict:
    """Runs `count` scouts in parallel on `model` and returns {role: [new candidate words]}."""
    n = max(0, min(count, len(SCOUT_ROLES)))
    if n == 0 or not history:
        return {}
    guessed = {h["guess"] for h in history} | set(rejected)
    ranked = sorted(history, key=lambda h: -h["similarity"])[:40]
    shown = ranked + [h for h in history[-15:] if h not in ranked]
    table = "\n".join("%s\t%.1f\t%s" % (h["guess"], h["similarity"],
                                          h["distance"] if h["distance"] and h["distance"] > 0 else "-")
                      for h in shown)
    ref = (f"Reference scores today: nearest {thresholds['nearest']}, 10th {thresholds['tenth']}, "
           f"1000th {thresholds['thousandth']}.\n" if thresholds else "")
    sub_args = argparse.Namespace(**vars(args))
    sub_args.model = model

    def one(role_and_text):
        role, text = role_and_text
        prompt = (f"{ref}Guesses so far ({len(history)}), format: word<TAB>similarity<TAB>rank:\n{table}\n\n"
                  f"{text}\nPropose 6 candidate words.")
        try:
            data = call_model(client, sub_args, SCOUT_SYSTEM, prompt, SCOUT_SCHEMA, "low")
        except Exception as e:  # a scout that fails is skipped; the solver still has its own judgement
            return role, {"note": f"failed: {type(e).__name__}", "candidates": []}
        words: list[str] = []
        for w in data.get("candidates", []):
            w = w.strip()
            if w and w not in guessed and w not in words:
                words.append(w)
        return role, {"note": data.get("note", ""), "candidates": words}

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = dict(pool.map(one, SCOUT_ROLES[:n]))
    emit({"type": "subagents", "model": model,
          "scouts": [{"role": r, "note": v["note"], "candidates": v["candidates"]} for r, v in results.items()]})
    return {r: v["candidates"] for r, v in results.items() if v["candidates"]}


# ---------- supervisor ----------

SUPERVISOR_SYSTEM = """You supervise an automatic Hebrew Semantle solver (another model plays; you only watch). Before
each round you set the player's budget:
- effort: low, medium, high, xhigh or max. How hard the player thinks: deeper reasoning, more time, more usage.
  Some models have no effort levels; then your effort choice is ignored.
- subagents: 0 to 3 scouts that explore in parallel before the player does (field-scout, neighbour-scout,
  triangulator). Every scout is one extra model call per round.
- scout_model: which model the scouts run on, picked from the list the user message gives you. A small model is
  fast and cheap; a stronger one proposes better words but is slower and costs much more per scout; a local model
  costs nothing but may be weak.
Spend where it helps and save where it does not.

Decide from the progress summary you are given:
- Early exploration with broad probes needs little thinking and little help: effort low or medium, and one or
  two scouts on a small model, or none.
- Raise effort when the player struggles: no new best score or rank for 3 or more rounds, guesses
  circling the same cluster, rank stuck at a low number, or similarity plateauing below the 1000th-nearest
  reference score. Raise one level for a mild stall, two for a long one. Use max only after 6+ rounds
  with no improvement.
- Scouts are the stronger tool against a plateau, because a stall is usually a strategy problem: the triangulator
  jumps to the outcome, cause or opposite of the top words. After 2 stalled rounds use all 3 scouts. After 4
  stalled rounds with 3 scouts on a small model, try a stronger model for them.
- When the player improves round after round, save: lower effort if it is high, and use fewer scouts or none.
- Never change anything without a reason you can state in one sentence.

# Learning from the record
The user message may include what you know from earlier games: how often each effort level and each scout
budget led to progress, how your earlier changes turned out, and lessons you wrote yourself after those games.
Use it.
- If raising effort rarely led to progress, do not lean on it; if a scout budget rarely did, do not lean on that.
- If one of your lessons says you acted too late, too early or pointlessly, act on it now.
- Your decisions in the current game are listed with what followed each one. Do not repeat a change that
  just failed to help.
- To learn what effort is worth, the program sometimes runs a round ONE OR TWO LEVELS ABOVE the effort you chose,
  at random, and marks it as an experiment. Those rounds are not your decisions. The statistics you are shown
  include them, so use them: if higher effort clearly helped, raise it sooner; if it made no difference, do not."""

REFLECT_SCHEMA = {
    "type": "object",
    "properties": {"effort_verdict": {"type": "string", "enum": ["too_low", "about_right", "too_high"]},
                   "scouts_verdict": {"type": "string", "enum": ["too_few", "about_right", "too_many"]},
                   "lessons": {"type": "array", "items": {"type": "string"}}},
    "required": ["effort_verdict", "scouts_verdict", "lessons"],
    "additionalProperties": False,
}

SUPERVISOR_SCHEMA = {
    "type": "object",
    "properties": {"effort": {"type": "string", "enum": EFFORTS},
                   "subagents": {"type": "integer", "enum": [0, 1, 2, 3]},
                   "scout_model": {"type": "string"},  # an enum of the models on offer is added per call
                   "reason": {"type": "string"}},
    "required": ["effort", "subagents", "scout_model", "reason"],
    "additionalProperties": False,
}

MODEL_NOTES = {"haiku": "smallest and fastest, cheapest", "sonnet": "balanced speed, cost and skill",
               "opus": "stronger, slower, costly", "fable": "most capable, slowest, most expensive"}


def supervisor_schema(models: list[str]) -> dict:
    schema = copy.deepcopy(SUPERVISOR_SCHEMA)
    schema["properties"]["scout_model"]["enum"] = models
    return schema


def scout_choices(args, local: list[dict]) -> list[tuple[str, str]]:
    """The models the supervisor may pick for the scouts, each with a plain note. The first is the default."""
    out: dict[str, str] = {}

    def add(model: str, note: str) -> None:
        if model and model != "auto" and model not in out:
            out[model] = note

    add("haiku" if args.subagent_model == "auto" else args.subagent_model, "the default scout model")
    add(args.model, "the player's own model")
    for alias in MODELS:
        add(alias, MODEL_NOTES[alias])
    for m in local:
        add(m["id"], "a local model on this computer: free to run, usually weaker")
    return list(out.items())


def experiment_note(r: dict) -> str:
    return f" [experiment: you chose {r['chosen_effort']}]" if r.get("experiment") else ""


def decision_text(log: list[dict]) -> str:
    """One line per supervisor decision with what the next round showed."""
    lines = []
    for d in log:
        out = d.get("outcome")
        result = "pending" if not out else ("improved the best score or rank" if out["improved"] else "no gain")
        change = (f"{d['previous']} -> {d['effort']}" if d["previous"] != d["effort"] else f"kept {d['effort']}"
                  ) if d.get("has_effort", True) else "no effort levels"
        scouts = ""
        if d.get("subagents") is not None:
            was = d.get("previous_subagents")
            on = d.get("scout_model") or d.get("scout_tier") or "default"
            scouts = (f", scouts {was} -> {d['subagents']}" if was is not None and was != d["subagents"]
                      else f", scouts {d['subagents']}") + f" on {on}"
            if d.get("previous_model") and d.get("scout_model") and d["previous_model"] != d["scout_model"]:
                scouts += f" (was {d['previous_model']})"
        lines.append(f"- after round {d['after_round']}: {change}{scouts} (stalled {d.get('stalled') or 0} rounds). "
                     f"Next round: {result}.")
    return "\n".join(lines)


def supervisor_digest(know: dict, game: dict) -> str:
    """What the supervisor learned from earlier games: effort and scout statistics, its changes, its lessons."""
    past = [g for g in know["games"] if g is not game]
    by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_scouts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    decisions = []
    for g in past:
        for r in g.get("rounds", []):
            if r.get("effort"):  # rounds of a model without effort levels say nothing about effort
                by[r["effort"]][0] += 1
                by[r["effort"]][1] += 1 if r["improved"] else 0
            b = r.get("budget")
            if b:
                key = f"{b['subagents']} scouts on {b.get('model') or b.get('tier')}" if b["subagents"] else "no scouts"
                by_scouts[key][0] += 1
                by_scouts[key][1] += 1 if r["improved"] else 0
        decisions += [d for d in g.get("supervisor_log", []) if d.get("outcome")]
    parts = []
    if by:
        order = sorted(by, key=lambda e: EFFORTS.index(e) if e in EFFORTS else len(EFFORTS))
        parts.append("Effort levels used in earlier games (rounds played, share of rounds that improved the best "
                     "score or rank): " + ", ".join(f"{e} {by[e][0]} rounds {100 * by[e][1] // by[e][0]}%" for e in order))
    if by_scouts:
        parts.append("Scout budgets used in earlier games (rounds played, share that improved the best score or "
                     "rank): " + ", ".join(f"{k} {n} rounds {100 * i // n}%" for k, (n, i) in sorted(by_scouts.items())))
    raises = [d for d in decisions if d["previous"] in EFFORTS and EFFORTS.index(d["effort"]) > EFFORTS.index(d["previous"])]
    if raises:
        parts.append(f"You raised effort {len(raises)} times; the very next round improved after "
                     f"{sum(1 for d in raises if d['outcome']['improved'])} of them.")
    more_scouts = [d for d in decisions if d.get("previous_subagents") is not None and d.get("subagents", 0) > d["previous_subagents"]]
    if more_scouts:
        parts.append(f"You added scouts {len(more_scouts)} times; the very next round improved after "
                     f"{sum(1 for d in more_scouts if d['outcome']['improved'])} of them.")
    exp = [r for g in past for r in g.get("rounds", []) if r.get("experiment")]
    if exp:
        base = [r for g in past for r in g.get("rounds", [])
                if r.get("effort") and not r.get("experiment") and r.get("stalled_before") == 0 and r.get("n", 0) >= 2]
        rate = lambda rs: f"{100 * sum(1 for r in rs if r['improved']) // len(rs)}%" if rs else "no data"
        parts.append(f"Effort experiments (random rounds above the effort you chose): {len(exp)} rounds, "
                     f"{rate(exp)} improved the best score or rank; comparable normal rounds (not the first, no "
                     f"stall): {len(base)}, {rate(base)} improved.")
    solved = [guess_count(g) for g in past if g.get("solved")]
    if solved:
        parts.append("Guesses needed in solved games: " + ", ".join(map(str, solved)) + ".")
    own = know.get("supervisor_lessons", [])[-10:]
    if own:
        parts.append("Your own lessons from earlier games (follow them):\n" + "\n".join(f"- {l}" for l in own))
    return ("What you know from earlier games:\n" + "\n".join(parts) + "\n\n") if parts else ""


def reflect_supervisor(client, args, game: dict, know: dict) -> list[str]:
    """After a game, the supervisor reviews its own decisions and writes lessons about its mistakes."""
    def budget(r):
        b = r.get("budget")
        return f", scouts {b['subagents']} on {b.get('model') or b.get('tier')}" if b else ""
    rounds = "\n".join(f"round {r['n']}: {'effort ' + r['effort'] if r.get('effort') else 'no effort levels'}{budget(r)}{experiment_note(r)}, "
                       f"best similarity {r['best_after']:.1f}, "
                       f"{'improved' if r['improved'] else 'no gain'}" for r in game.get("rounds", []))
    earlier = know.get("supervisor_lessons", [])[-10:]
    prompt = (
        f"The game is over: solved in {len(game['guesses'])} guesses over {len(game.get('rounds', []))} rounds.\n"
        f"Rounds:\n{rounds}\n\nYour decisions (effort and scouts) and what followed:\n"
        f"{decision_text(game.get('supervisor_log', [])) or '(none)'}\n\n"
        f"Lessons you already wrote:\n{chr(10).join('- ' + l for l in earlier) or '(none yet)'}\n\n"
        "First judge the whole game, the way the person running the solver would: was the effort you chose too low, "
        "about right or too high, and was the scout budget too few, about right or too many (answer about_right "
        "when the user fixed the scouts). Then write 2-3 NEW one-sentence lessons about your decisions: moments you "
        "changed effort or the scout budget too late, too early or pointlessly, and what to do differently. If a "
        "setting made no visible difference to the progress, say so plainly. Do not repeat a lesson you already wrote."
    )
    sup_args = argparse.Namespace(**vars(args))
    sup_args.model = args.supervisor
    data = call_model(client, sup_args, SUPERVISOR_SYSTEM, prompt, REFLECT_SCHEMA, "low")
    verdict = (f"Self-assessment of a game: effort {data['effort_verdict'].replace('_', ' ')}, "
               f"scouts {data['scouts_verdict'].replace('_', ' ')}.")
    return [verdict] + data["lessons"]


def stall_rounds(best_per_round: list[float]) -> int:
    """Rounds since the best similarity last improved."""
    best, last = float("-inf"), 0
    for i, b in enumerate(best_per_round):
        if b > best:
            best, last = b, i
    return len(best_per_round) - 1 - last if best_per_round else 0


class Supervisor:
    """Reviews the game in a background thread while the main model thinks. A verdict is applied at the
    start of the next round, so the review never adds latency. A verdict is the budget for that round:
    {"effort", "subagents", "model"} (the model the scouts run on)."""

    def __init__(self, args, client, emit, know: dict, game: dict):
        self.args, self.client, self.emit = args, client, emit
        self.know, self.game = know, game
        self.digest = supervisor_digest(know, game)  # fixed for the game; this game's decisions are added per review
        self.model = args.supervisor
        try:  # local models are listed once per game; this is quick and empty when no local server runs
            self.local = providers.local_models() if args.subagent_model == "auto" else []
        except Exception:
            self.local = []
        self.thread: threading.Thread | None = None
        self.verdict: dict | None = None

    def maybe_experiment(self, effort: str, stalled: int, done: int) -> str | None:
        """An effort level above `effort` to try this round, or None. Only on quiet rounds (no stall, which is when
        the supervisor would be raising effort for real), never above the ceiling, never more than twice a game,
        and only for levels with too few recorded rounds to say what they are worth."""
        if done >= MAX_EXPERIMENTS or stalled > 0 or effort not in EFFORTS:
            return None
        above = EFFORTS[EFFORTS.index(effort) + 1: EFFORTS.index(EXPERIMENT_CEILING) + 1]
        counts: dict[str, int] = defaultdict(int)
        for g in self.know["games"]:  # earlier games and this one: `know` is the view that includes the current game
            for r in g.get("rounds", []):
                if r.get("effort"):
                    counts[r["effort"]] += 1
        need = [e for e in above if counts[e] < EXPERIMENT_SAMPLES]
        if not need or rng.random() >= EXPERIMENT_CHANCE:
            return None
        return min(need, key=lambda e: (counts[e], EFFORTS.index(e)))

    def _models_text(self, budget: dict) -> str:
        fixed = [what for what, is_fixed in (("the number of scouts", self.args.subagents != "auto"),
                                             ("the scout model", self.args.subagent_model != "auto")) if is_fixed]
        lines = "\n".join(f"- {m}: {note}" for m, note in scout_choices(self.args, self.local))
        return ("Models you may pick for the scouts:\n" + lines + "\n"
                + (f"Fixed by the user, you cannot change: {' and '.join(fixed)}.\n" if fixed else ""))

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def take(self) -> dict | None:
        """The budget the supervisor chose since the last call, or None."""
        verdict, self.verdict = self.verdict, None
        return verdict

    def _ask(self, prompt: str, effort: str, budget: dict, stalled: int) -> dict:
        """One supervisor call; falls back to a simple rule when it fails. Returns the verdict plus its reason."""
        sup_args = argparse.Namespace(**vars(self.args))
        sup_args.model = self.model
        names = [m for m, _ in scout_choices(self.args, self.local)]
        try:
            data = call_model(self.client, sup_args, SUPERVISOR_SYSTEM, prompt, supervisor_schema(names), "low")
            new = {"effort": data["effort"], "subagents": data["subagents"],
                   "model": data["scout_model"] if data["scout_model"] in names else budget["model"],
                   "reason": data["reason"]}
        except Exception as e:  # a failed review never stops the game
            new = {"effort": EFFORTS[min(EFFORTS.index(effort) + 1, len(EFFORTS) - 1)] if stalled >= 3 else effort,
                   "subagents": 3 if stalled >= 2 else budget["subagents"], "model": budget["model"],
                   "reason": f"supervisor call failed ({type(e).__name__}); rule: more scouts after 2 stalled rounds, "
                             "one effort level up after 3"}
        if not effort_ok(self.args.model):
            new["effort"] = effort  # the player's model has no effort levels, so only the scouts can change
        if self.args.subagents != "auto":
            new["subagents"] = budget["subagents"]  # the user fixed the number of scouts
        if self.args.subagent_model != "auto":
            new["model"] = budget["model"]  # ...or their model
        return new

    def _announce(self, new: dict, effort: str, budget: dict, stalled: int, previous_effort: str) -> None:
        self.verdict = {"effort": new["effort"], "subagents": new["subagents"], "model": new["model"]}
        self.emit({"type": "supervisor", "effort": new["effort"], "previous": previous_effort, "reason": new["reason"],
                   "has_effort": effort_ok(self.args.model),
                   "stalled": stalled, "subagents": new["subagents"], "previous_subagents": budget["subagents"],
                   "scout_model": new["model"], "previous_model": budget["model"]})

    def choose_start(self, know: dict, thresholds: dict | None, budget: dict) -> dict:
        """Picks the first round's budget. Blocks for one short call."""
        past = [guess_count(g) for g in know["games"] if g.get("solved")]
        prompt = (
            f"{self.digest}A new game is starting: no guesses yet.\n"
            f"Main model: {self.args.model} (effort levels: {'yes' if effort_ok(self.args.model) else 'no'}).\n"
            f"Scouts start as {budget['subagents']} on {budget['model']}.\n{self._models_text(budget)}"
            + (f"Past solved games, guesses needed: {', '.join(map(str, past))}.\n" if past
               else "No past games yet.\n")
            + (f"Reference scores today: nearest {thresholds['nearest']}, 10th {thresholds['tenth']}, "
               f"1000th {thresholds['thousandth']}.\n" if thresholds else "")
            + "\nWhat budget should the player have for the first rounds?"
        )
        new = self._ask(prompt, "medium", budget, 0)
        if not effort_ok(self.args.model):
            new["effort"] = "medium"
        self._announce(new, "medium", budget, 0, "auto")
        return {"effort": new["effort"], "subagents": new["subagents"], "model": new["model"]}

    def review(self, history: list[dict], best_per_round: list[float], effort: str,
               thresholds: dict | None, budget: dict) -> None:
        if self.busy():
            return
        snapshot = [dict(h) for h in history]
        self.thread = threading.Thread(
            target=self._run, args=(snapshot, list(best_per_round), effort, thresholds, dict(budget)), daemon=True)
        self.thread.start()

    def _run(self, history, best_per_round, effort, thresholds, budget) -> None:
        stalled = stall_rounds(best_per_round)
        ranked = [h for h in history if h["distance"] and 0 < h["distance"] < 1000]
        best_rank = max((h["distance"] for h in ranked), default=None)
        top = ", ".join("%s %.1f" % (h["guess"], h["similarity"])
                        for h in sorted(history, key=lambda h: -h["similarity"])[:8])
        mine = decision_text(list(self.game.get("supervisor_log", [])))
        prompt = (
            f"{self.digest}"
            + (f"Your decisions so far in this game:\n{mine}\n\n" if mine else "")
            + f"Rounds played: {len(best_per_round)}. Guesses: {len(history)}. Current effort: {effort}"
            f" (the player's model {'has' if effort_ok(self.args.model) else 'has no'} effort levels). "
            f"Current scouts: {budget['subagents']} on {budget['model']}.\n{self._models_text(budget)}"
            + f"Best similarity after each round: {', '.join(f'{b:.1f}' for b in best_per_round)}.\n"
            f"Rounds since the best similarity last improved: {stalled}.\n"
            f"Best rank so far: {best_rank if best_rank is not None else 'none yet'}. "
            f"Guesses with a rank: {len(ranked)}.\n"
            + (f"Reference scores today: nearest {thresholds['nearest']}, 10th {thresholds['tenth']}, "
               f"1000th {thresholds['thousandth']}.\n" if thresholds else "")
            + f"Top guesses: {top}.\n\n"
            "What budget should the player have for the next round?"
        )
        new = self._ask(prompt, effort, budget, stalled)
        self._announce(new, effort, budget, stalled, effort)


# ---------- game ----------

def play(args, emit, stop: threading.Event | None = None, live: dict | None = None) -> int:
    """Plays today's puzzle. Reports progress by calling emit(event_dict); returns a process exit code.

    `live` is a dict the caller may change while the game runs ("model", "supervisor"); the new values
    apply from the next round.
    Event types: start, waiting, reasoning, supervisor, model, guess, rejected, found, lessons, supervisor_lessons, subagents, stopped, error.
    """
    stop = stop or threading.Event()
    client = None  # the Anthropic client is created on first use (see call_model)
    know = load_knowledge(args.knowledge)
    puzzle, thresholds = get_puzzle_info()

    # Test mode reads everything from knowledge.json but writes only to the test file, which is wiped at the
    # start of every test run.
    out_path = args.knowledge
    fresh_sup: list[str] = []  # what a test run adds, kept apart so the test file holds only that
    fresh_rej: list[str] = []
    if args.test:
        out_path = args.test_file
        out_path.unlink(missing_ok=True)

    # Whatever happened earlier today is ignored, however the run is started: no resuming an unfinished game,
    # no skipping a solved puzzle, and earlier records of today's puzzle (and the supervisor lessons they wrote)
    # stay out of what the solver reads, or it would "remember" the secret. The records remain in the file;
    # only `view`, the copy the solver reads, is filtered.
    game = {"puzzle": puzzle, "date": date.today().isoformat(), "model": args.model,
            "solved": False, "secret": None, "guesses": [], "lessons": [],
            # the setup, so results can be compared across days
            "config": {"model": args.model, "supervisor": args.supervisor, "subagents": args.subagents,
                       "subagent_model": args.subagent_model}}
    earlier_today = [g for g in know["games"] if puzzle is not None and g.get("puzzle") == puzzle]
    know["games"].append(game)
    skip = {id(g) for g in earlier_today}
    written = {l for g in earlier_today for l in g.get("supervisor_lessons_written", [])}
    view = {**know, "games": [g for g in know["games"] if id(g) not in skip],
            "supervisor_lessons": [l for l in know.get("supervisor_lessons", []) if l not in written]}
    game["thresholds"] = thresholds
    def save() -> None:  # games are stored as counts and a summary, never every guess
        save_knowledge(out_path, {"version": 1, "test_run": True, "games": [compact_game(game)],
                                  "supervisor_lessons": fresh_sup, "rejected": fresh_rej} if args.test
                       else {**know, "games": [compact_game(g) if g is game else g for g in know["games"]]})

    rounds = game.setdefault("rounds", [])              # one record per round: effort, progress, scout usage
    sup_log = game.setdefault("supervisor_log", [])     # every supervisor decision, with its outcome filled in later
    raw_emit = emit

    def emit(ev: dict) -> None:  # the supervisor's decisions are also kept in the game record
        if ev["type"] == "supervisor":
            sup_log.append({"after_round": len(rounds), "previous": ev["previous"], "effort": ev["effort"],
                            "reason": ev["reason"], "stalled": ev.get("stalled"), "has_effort": ev.get("has_effort", True),
                            "subagents": ev.get("subagents"), "previous_subagents": ev.get("previous_subagents"),
                            "scout_model": ev.get("scout_model"), "previous_model": ev.get("previous_model"),
                            "outcome": None})
        raw_emit(ev)

    history = game["guesses"]
    resumed = len(history)
    seen = {h["guess"] for h in history} | set(know["rejected"])
    rejected: list[str] = []
    memory = build_memory(view, game)

    emit({"type": "start", "puzzle": puzzle, "model": args.model, "effort": "auto" if effort_ok(args.model) else None,
          "backend": args.backend, "supervisor": args.supervisor, "subagents": args.subagents, "test": args.test, "past_games": len(view["games"]) - 1, "thresholds": thresholds,
          "resumed": resumed})
    for i, h in enumerate(history, 1):
        emit({"type": "guess", "n": i, "word": h["guess"], "similarity": h["similarity"],
              "distance": h["distance"], "resumed": True})

    limit = args.max_guesses or float("inf")
    stalls = 0
    effort = "auto"  # effort is never set by hand: the supervisor picks it (medium when the supervisor is off)
    best_per_round: list[float] = [r["best_after"] for r in rounds]
    supervisor: Supervisor | None = None
    # A number of scouts chosen by the user is fixed; with "auto" the supervisor manages it, starting at 3.
    budget = {"subagents": 3 if args.subagents == "auto" else int(args.subagents),
              "model": "haiku" if args.subagent_model == "auto" else args.subagent_model}
    live = live if live is not None else {}
    try:
        while len(history) < limit and stalls < STALL_LIMIT and not stop.is_set():
            # Model and supervisor can be switched while the game runs; the change applies from this round.
            if live.get("model") and live["model"] != args.model:
                emit({"type": "model", "kind": "main", "model": live["model"], "previous": args.model})
                args.model = live["model"]
            if live.get("supervisor") and live["supervisor"] != args.supervisor:
                emit({"type": "model", "kind": "supervisor", "model": live["supervisor"],
                      "previous": args.supervisor})
                args.supervisor = live["supervisor"]
                supervisor = None
            # The supervisor sets the budget: effort (when the model has effort levels), how many scouts, and their model.
            if args.supervisor == "off":
                supervisor = None
            elif supervisor is None:
                supervisor = Supervisor(args, client, emit, view, game)
            if effort == "auto":  # the supervisor picks the starting budget; without one, medium and the defaults
                if supervisor:
                    v = supervisor.choose_start(view, thresholds, budget)
                    effort = v["effort"] if effort_ok(args.model) else "medium"
                    if args.subagents == "auto":
                        budget["subagents"] = v["subagents"]
                    if args.subagent_model == "auto":
                        budget["model"] = v["model"]
                else:
                    effort = "medium"
            if supervisor:
                v = supervisor.take()  # verdict from the review that ran during the last round
                if v:
                    if effort_ok(args.model):
                        effort = v["effort"]
                    if args.subagents == "auto":
                        budget["subagents"] = v["subagents"]
                    if args.subagent_model == "auto":
                        budget["model"] = v["model"]
                if history:
                    supervisor.review(history, best_per_round, effort, thresholds, budget)
            scout_model = budget["model"]
            stalled_now = stall_rounds(best_per_round)
            tried = None  # an effort experiment: this round runs above the supervisor's choice
            if supervisor and args.experiments and effort_ok(args.model) and history:
                tried = supervisor.maybe_experiment(effort, stalled_now, sum(1 for r in rounds if r.get("experiment")))
            round_effort = tried or effort
            if tried:
                emit({"type": "experiment", "chosen": effort, "tried": tried})
            levels = effort_ok(args.model)  # models without effort levels (Haiku, most local models) show none
            shown_effort = round_effort if levels else None
            emit({"type": "waiting", "effort": shown_effort, "model": args.model, "budget": dict(budget)})
            scouts = run_scouts(client, args, history, rejected, thresholds, emit, budget["subagents"], scout_model)
            prev_best = max((h["similarity"] for h in history), default=None)
            prev_rank = max((h["distance"] for h in history if h["distance"] and h["distance"] > 0), default=0)
            accepted: list[dict] = []
            data = call_model(client, args, SYSTEM,
                               build_prompt(memory, history, rejected, thresholds, scouts),
                               SCHEMA, round_effort)
            emit({"type": "reasoning", "text": data["reasoning"], "guesses": data["guesses"],
                  "effort": shown_effort, "model": args.model, "budget": dict(budget), "thinking_tokens": (data.get("_meta") or {}).get("thinking_tokens")})
            progressed = False
            for word in data["guesses"][:MAX_GUESSES]:  # the model sets the size; this only caps a runaway reply
                if stop.is_set():
                    break
                word = word.strip()
                if not word or word in seen:
                    continue
                seen.add(word)
                rec = get_distance(word)
                if rec is None or rec.get("similarity") is None:
                    rejected.append(word)
                    know["rejected"].append(word)
                    fresh_rej.append(word)
                    emit({"type": "rejected", "word": word})
                    continue
                progressed = True
                history.append({"guess": word, "similarity": rec["similarity"], "distance": rec["distance"]})
                accepted.append(history[-1])
                save()
                emit({"type": "guess", "n": len(history), "word": word, "similarity": rec["similarity"],
                      "distance": rec["distance"]})
                if rec["distance"] == 1000:
                    game["solved"], game["secret"] = True, word
                    break
            if history:
                best_after = max(h["similarity"] for h in history)
                rank_after = max((h["distance"] for h in history if h["distance"] and h["distance"] > 0), default=0)
                improved = prev_best is None or best_after > prev_best or rank_after > prev_rank
                scout_use = {}
                for role, words in scouts.items():  # what the scout's words were worth, as counts
                    taken = [h for h in accepted if h["guess"] in words]
                    scout_use[role] = {"proposed": len(words), "used": len(taken),
                                       "sum_sim": round(sum(h["similarity"] for h in taken), 2),
                                       "ranked": sum(1 for h in taken if h["distance"] and h["distance"] > 0)}
                rounds.append({"n": len(rounds) + 1, "model": args.model, "effort": shown_effort, "improved": improved,
                               "best_after": best_after, "rank_after": rank_after,
                               "size": len(accepted), "asked": len(data["guesses"]),
                               "experiment": bool(tried), "chosen_effort": effort if tried else None,
                               "stalled_before": stalled_now,
                               "thinking_tokens": (data.get("_meta") or {}).get("thinking_tokens"),
                               "budget": {"subagents": budget["subagents"], "model": scout_model},
                               "scouts": scout_use})
                for d in sup_log:  # the first decision at this effort that has not been judged yet gets its outcome
                    # an experiment round says nothing about the supervisor's decision, so it is not judged
                    if (d["outcome"] is None and d["effort"] == effort and d["after_round"] < rounds[-1]["n"]
                            and not tried):
                        d["outcome"] = {"round": rounds[-1]["n"], "improved": improved,
                                        "gain": round(best_after - (prev_best if prev_best is not None else best_after), 2)}
                        break
                best_per_round.append(best_after)
            if game["solved"]:
                break
            stalls = 0 if progressed else stalls + 1
    except KeyboardInterrupt:
        save()
        emit({"type": "stopped", "reason": "interrupted", "best": best_guess(history)})
        return 130
    except (*API_ERRORS, RuntimeError, subprocess.TimeoutExpired, requests.RequestException) as e:
        save()
        emit({"type": "error", "message": str(e)})
        return 2

    if game["solved"]:
        emit({"type": "found", "secret": game["secret"], "guesses": len(history)})
        summarise(client, args, game, view, emit, solved=True)
        if sup_log and args.supervisor != "off":
            try:  # the supervisor reviews its own effort decisions and keeps lessons about its mistakes
                new = reflect_supervisor(client, args, game, view)
                game["supervisor_lessons_written"] = new  # so a later run on this puzzle can leave them out
                know["supervisor_lessons"] = (know.get("supervisor_lessons", []) + new)[-30:]
                fresh_sup.extend(new)
                emit({"type": "supervisor_lessons", "lessons": new})
            except Exception as e:
                emit({"type": "error", "message": f"supervisor could not write lessons: {e}"})
        save()
        return 0

    reason = ("stopped by user" if stop.is_set()
              else "no new words proposed" if stalls >= STALL_LIMIT else "guess limit reached")
    if not stop.is_set() and len(history) >= 5:  # a game that ran out is worth a summary too; a stop is not
        summarise(client, args, game, view, emit, solved=False)
    save()
    emit({"type": "stopped", "reason": reason, "best": best_guess(history)})
    return 1


def summarise(client, args, game: dict, view: dict, emit, solved: bool) -> None:
    """Writes the game's summary and lessons. A failure here never loses the game."""
    try:
        res = write_lessons(client, args, game, view, solved)
        game["summary"], game["lessons"] = res["summary"], res["lessons"]
        emit({"type": "lessons", "lessons": game["lessons"], "summary": game["summary"]})
    except Exception as e:
        emit({"type": "error", "message": f"could not write the summary: {e}"})


def best_guess(history: list[dict]) -> str | None:
    return max(history, key=lambda h: h["similarity"])["guess"] if history else None


# ---------- command line ----------

def make_cli_emit(verbose: bool):
    def emit(ev: dict) -> None:
        t = ev["type"]
        if t == "start":
            resumed = f" | resuming {ev['resumed']} saved guesses" if ev["resumed"] else ""
            print(f"Puzzle {ev['puzzle']} | model: {ev['model']} | "
                  f"{'effort: ' + ev['effort'] + ' | ' if ev.get('effort') else ''}"
                  f"past games: {ev['past_games']}{resumed}")
        elif t == "reasoning" and verbose:
            print(f"  [{ev.get('model', 'model')}{'/' + ev['effort'] if ev.get('effort') else ''}] {ev['text']}")
        elif t == "supervisor":
            change = ((f"{ev['previous']} -> {ev['effort']}" if ev["effort"] != ev["previous"] else f"keep {ev['effort']}")
                      if ev.get("has_effort", True) else "no effort levels")
            scouts = ""
            if ev.get("subagents") is not None:
                scouts = f", scouts {ev['previous_subagents']} -> {ev['subagents']} on {ev['scout_model']}"
            print(f"  [supervisor] {change}{scouts}: {ev['reason']}")
        elif t == "guess" and not ev.get("resumed"):
            rank = ev["distance"]
            tag = f"{rank}/1000" if rank and rank > 0 else "far"
            print(f"#{ev['n']:>3} {ev['word']}\t{ev['similarity']:6.2f}\t{tag}")
        elif t == "rejected":
            print(f"  {ev['word']}: not in vocabulary")
        elif t == "found":
            print(f"\nFound the secret word: {ev['secret']} in {ev['guesses']} guesses")
        elif t == "lessons":
            print(f"  summary: {ev.get('summary', '')}")
            for lesson in ev["lessons"]:
                print(f"  lesson: {lesson}")
        elif t == "experiment":
            print(f"  [experiment] effort {ev['chosen']} chosen, this round runs at {ev['tried']} to gather data")
        elif t == "supervisor_lessons":
            for lesson in ev["lessons"]:
                print(f"  supervisor lesson: {lesson}")
        elif t == "subagents":
            for sc in ev["scouts"]:
                print(f"  [{sc['role']}] {', '.join(sc['candidates']) or '(nothing)'}")
        elif t == "stopped":
            print(f"\nStopped ({ev['reason']}). Best: {ev['best'] or '-'}. The game is saved; running again starts a new one.")
        elif t == "error":
            print(f"\nThe model call failed: {ev['message']}\nThe game is saved; running again starts a new one.")
    return emit


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="sonnet",
                   help=f"alias ({', '.join(MODELS)}: always the newest of that family) or any full model ID, "
                        "e.g. claude-sonnet-5 (default: sonnet)")
    p.add_argument("--supervisor", default="haiku",
                   help="model that watches the game in the background and adjusts the solver's effort "
                        "(alias or model ID; 'off' disables; no effect when the solver itself is Haiku)")
    p.add_argument("--no-experiments", dest="experiments", action="store_false",
                   help="turn off effort experiments (by default a round sometimes runs one or two effort levels "
                        "above the supervisor's choice, at random, to learn what effort is worth)")
    p.add_argument("--subagents", default="auto", choices=["auto", "0", "1", "2", "3"],
                   help="scouts that explore in parallel each round and feed the solver proposals "
                        "(field-scout, neighbour-scout, triangulator). A number is fixed for the whole game "
                        "(0 = none). auto (default): the supervisor decides each round, starting at 3")
    p.add_argument("--subagent-model", default="auto",
                   help="model the scouts run on (alias, model ID or provider:model). A model is fixed for the whole "
                        "game. auto (default): the supervisor picks one each round from the available models, "
                        "starting with haiku")
    p.add_argument("--backend", default="auto", choices=["auto", "cli", "api"],
                   help="how to reach each provider. cli: the provider's own CLI signed in with your "
                        "subscription (Claude Code, Codex, Gemini CLI), no key. api: an API key. "
                        "auto (default): the CLI when installed, otherwise the key. Local servers "
                        "(Ollama, LM Studio) ignore this")
    p.add_argument("--claude-bin", help="path to the Claude Code binary (cli backend; auto-detected)")
    p.add_argument("--max-guesses", type=int, default=0,
                   help="stop after this many accepted guesses (default 0 = no limit)")
    p.add_argument("--knowledge", type=Path, default=KNOWLEDGE_PATH, help="path of the knowledge JSON")
    p.add_argument("--test", action="store_true",
                   help="test run: read memory from --knowledge but write only to --test-file, which is wiped at "
                        "the start of every test run; --knowledge is never modified")
    p.add_argument("--test-file", type=Path, default=Path(__file__).with_name("knowledge_test.json"),
                   help="where a --test run writes (default knowledge_test.json)")
    p.add_argument("--races-file", type=Path, default=Path(__file__).with_name("knowledge_races.json"),
                   help="where race results are kept (default knowledge_races.json)")
    p.add_argument("--verbose", "-v", action="store_true", help="print Claude's reasoning")
    p.add_argument("--ui", action="store_true", help="serve a live web UI and open it in the browser")
    p.add_argument("--autostart", action="store_true", help="with --ui: start solving immediately")
    p.add_argument("--port", type=int, default=8765, help="with --ui: first port to try (default 8765)")
    p.add_argument("--no-browser", action="store_true", help="with --ui: do not open the browser")
    return p


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Hebrew output on Windows consoles (cp1252 by default)
    parser = build_parser()
    args = parser.parse_args()
    cli_emit = make_cli_emit(args.verbose)
    if args.ui:
        from ui_server import serve
        return serve(args, {"play": play, "emit": cli_emit, "model_options": model_options,
                            "provider_status": providers.provider_status,
                            "load_knowledge": lambda: knowledge_summary(load_knowledge(args.knowledge),
                                                                       load_races(args.races_file)),
                            "save_race": lambda race: append_race(args.races_file, race)})
    return play(args, cli_emit)


if __name__ == "__main__":
    sys.exit(main())
