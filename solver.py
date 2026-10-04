"""Semantle Word Hunter: plays Hebrew Semantle (https://semantle.ishefi.com) for you, driven by an AI model.

Each round the model sees every guess so far (sorted by similarity) plus a digest of what earlier games
taught it, and proposes a batch of new Hebrew words. The script submits them to the site's
/api/distance endpoint and loops until the secret word is found (distance == 1000).

Everything learned is stored in knowledge.json: every game's guesses, the secret word, words the game
rejected, and short lessons Claude writes after each game. The next run feeds a digest of that file
back into the prompt, so the solver improves over time. Progress is saved after every guess, so an
interrupted game resumes where it stopped.

Run it in the terminal (python solver.py) or with a live web UI (python solver.py --ui).
"""

import argparse
import json
import os
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
FEEDBACK_PREFIX = "USER FEEDBACK"
EXAMPLE_COUNT = 3  # worked examples (best past solves) included in every prompt
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
The user message may start with "Knowledge from previous games": past secret words, full worked examples
of earlier solves, which words were close to which secrets, strong broad probes, and lessons. Use it:
- Study the worked examples: they show how a real search moved from field to answer. Imitate what
  worked, avoid what wasted guesses.
- Reuse the strong broad probes when exploring, if they are not already in this game's history.
- Today's secret is a different word from every past secret. Past secrets are only evidence about how
  the embedding behaves, never candidates.
- Lessons are guidance from small samples: follow them, but let this game's actual scores override them.

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
suggest, why these words) and the "guesses" list with exactly the requested number of words."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "guesses": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "guesses"],
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
    tmp.replace(path)


def format_path(guesses: list[dict], head: int = 8, tail: int = 12) -> str:
    """Compact one-guess-per-line trail; long games keep the opening and the final approach."""
    rows = [f"{i}. {x['guess']}  {x['similarity']:.1f}  {x['distance'] if x['distance'] and x['distance'] > 0 else '-'}"
            for i, x in enumerate(guesses, 1)]
    if len(rows) > head + tail:
        rows = rows[:head] + [f"   ... ({len(rows) - head - tail} guesses omitted) ..."] + rows[-tail:]
    return "\n".join(rows)


def build_memory(know: dict, current: dict) -> str:
    """Digest of earlier games for the prompt: past secrets, worked examples, neighbours, probes, lessons."""
    past = [g for g in know["games"] if g is not current]
    if not past and not know.get("lessons"):
        return ""
    parts = []

    # A replayed puzzle appears twice; keep only the most efficient solve per secret word.
    best: dict[str, dict] = {}
    for g in past:
        if g.get("solved") and (g["secret"] not in best or len(g["guesses"]) < len(best[g["secret"]]["guesses"])):
            best[g["secret"]] = g
    solved = [g for g in past if g.get("solved") and best[g["secret"]] is g]
    if solved:
        parts.append("Past secret words (never the answer again, probably): "
                     + ", ".join(f"{g['secret']} ({len(g['guesses'])} guesses)" for g in solved[-30:]))
        assoc = []
        for g in solved[-10:]:
            near = sorted((x for x in g["guesses"] if x["distance"] and 0 < x["distance"] < 1000),
                          key=lambda x: -x["distance"])[:8]
            if near:
                assoc.append(f"  {g['secret']}: " + ", ".join(f"{x['guess']}({x['distance']})" for x in near))
        if assoc:
            parts.append("Words that ranked closest to past secrets (word(rank/1000)) - shows what "
                         "associations the embedding makes:\n" + "\n".join(assoc))

    # Worked examples: the most efficient solves, shown as the actual search path.
    for g in sorted(solved, key=lambda g: len(g["guesses"]))[:EXAMPLE_COUNT]:
        parts.append(f"Worked example - secret '{g['secret']}', solved in {len(g['guesses'])} guesses "
                     f"(word, similarity, rank):\n{format_path(g['guesses'])}")

    stats = defaultdict(list)
    for g in past:
        for x in g["guesses"]:
            stats[x["guess"]].append(x["similarity"])
    probes = sorted(((sum(v) / len(v), w, len(v)) for w, v in stats.items() if len(v) >= 2), reverse=True)[:20]
    if probes:
        parts.append("Words that scored highest on average across several games (good broad probes): "
                     + ", ".join(f"{w} ({m:.1f})" for m, w, _ in probes))

    # What each sub-agent's words were worth, from the rounds recorded in earlier games.
    track: dict[str, list] = defaultdict(lambda: [0, 0.0, 0])
    for g in past:
        for r in g.get("rounds", []):
            for role, info in (r.get("scouts") or {}).items():
                for _, sim, rank in info.get("used", []):
                    track[role][0] += 1
                    track[role][1] += sim
                    track[role][2] += 1 if rank and rank > 0 else 0
    used = [f"{role}: {n} words used, average similarity {total / n:.1f}, {ranked} with a rank"
            for role, (n, total, ranked) in track.items() if n]
    if used:
        parts.append("Sub-agent track record from earlier games (words you took from each scout): " + "; ".join(used))

    lessons = list(know.get("lessons", [])) + [l for g in past[-10:] for l in g.get("lessons", [])]
    if lessons:
        parts.append("Lessons from earlier games:\n" + "\n".join(f"- {l}" for l in lessons[-25:]))

    return "Knowledge from previous games:\n" + "\n\n".join(parts) + "\n\n"


def knowledge_summary(know: dict) -> dict:
    """What the web UI shows in its memory panel."""
    games = [{"puzzle": g.get("puzzle"), "date": g.get("date"), "secret": g.get("secret"),
              "solved": g.get("solved"), "guesses": len(g["guesses"]), "model": g.get("model")}
             for g in know["games"]]
    lessons = (know.get("lessons", []) + [l for g in know["games"][-10:] for l in g.get("lessons", [])])[-12:]
    return {"games": games[-30:], "lessons": lessons, "rejected": len(know["rejected"])}


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


def build_prompt(memory: str, history: list[dict], rejected: list[str], batch: int,
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
        f"Propose exactly {batch} new Hebrew words to guess next. "
        "Keep reasoning short (a few sentences)."
    )


def write_lessons(client, args, game: dict, know: dict) -> list[str]:
    earlier = (list(know.get("lessons", [])) + [l for g in know["games"] if g is not game for l in g.get("lessons", [])])[-25:]
    prompt = (
        f"You just solved a Hebrew Semantle puzzle in {len(game['guesses'])} guesses. "
        f"The secret word was: {game['secret']}.\n"
        f"Your guesses in order (word, similarity, rank):\n{format_path(game['guesses'], head=60, tail=60)}\n\n"
        f"Lessons already recorded from earlier games:\n"
        f"{chr(10).join('- ' + l for l in earlier) or '(none yet)'}\n\n"
        "Write 2-4 NEW short lessons for solving future puzzles faster, each one sentence: which probes "
        "were wasteful or useful, how quickly the right field was identified, which kind of association "
        "finally led to the secret. Do not repeat an existing lesson; if this game contradicts one, say so "
        "explicitly. One game is a small sample: phrase lessons about the search process (what to try "
        "when), not about what kind of word secrets usually are."
    )
    return call_model(client, args, SYSTEM, prompt, LESSON_SCHEMA, "low")["lessons"]


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


def run_scouts(client, args, history: list[dict], rejected: list[str], thresholds: dict | None, emit) -> dict:
    """Runs the scouts in parallel on a cheaper model and returns {role: [new candidate words]}."""
    n = max(0, min(args.subagents, len(SCOUT_ROLES)))
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
    sub_args.model = args.subagent_model

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
    emit({"type": "subagents", "model": args.subagent_model,
          "scouts": [{"role": r, "note": v["note"], "candidates": v["candidates"]} for r, v in results.items()]})
    return {r: v["candidates"] for r, v in results.items() if v["candidates"]}


# ---------- supervisor ----------

SUPERVISOR_SYSTEM = """You supervise an automatic Hebrew Semantle solver (another Claude model plays; you
only watch). Before each round you choose how hard the player should think, as an effort level:
low, medium, high, xhigh or max. Higher effort means deeper reasoning, more time and more usage.

Decide from the progress summary you are given:
- Early exploration with broad probes needs little thinking: low or medium.
- Raise effort when the player struggles: no new best score or rank for 3 or more rounds, guesses
  circling the same cluster, rank stuck at a low number, or similarity plateauing below the 1000th-nearest
  reference score. Raise one level for a mild stall, two for a long one. Use max only after 6+ rounds
  with no improvement.
- When the player is improving round after round, keep the current level, or lower it if the level is
  high and progress is easy.
- Never change effort without a reason you can state in one sentence.

# Learning from the record
The user message may include what you know from earlier games: how often each effort level led to progress,
how your earlier raises turned out, and lessons you wrote yourself after those games. Use it.
- If raising effort rarely led to progress, do not lean on it. A stall is often a strategy problem (the
  player circling one cluster of near-synonyms) and not a thinking-depth problem. The player has sub-agents
  that explore other axes, so a stall does not always call for more effort.
- If one of your lessons says you acted too late, too early or pointlessly, act on it now.
- Lines that start with USER FEEDBACK come from the person running the solver. They outrank your own
  lessons: follow them.
- Your decisions in the current game are listed with what followed each one. Do not repeat a change that
  just failed to help."""

SUPERVISOR_SCHEMA = {
    "type": "object",
    "properties": {"effort": {"type": "string", "enum": EFFORTS}, "reason": {"type": "string"}},
    "required": ["effort", "reason"],
    "additionalProperties": False,
}


def decision_text(log: list[dict]) -> str:
    """One line per supervisor decision with what the next round showed."""
    lines = []
    for d in log:
        out = d.get("outcome")
        result = "pending" if not out else ("improved the best score or rank" if out["improved"] else "no gain")
        change = f"{d['previous']} -> {d['effort']}" if d["previous"] != d["effort"] else f"kept {d['effort']}"
        lines.append(f"- after round {d['after_round']}: {change} (stalled {d.get('stalled') or 0} rounds). "
                     f"Next round: {result}.")
    return "\n".join(lines)


def supervisor_digest(know: dict, game: dict) -> str:
    """What the supervisor learned from earlier games: effort statistics, its own raises, its own lessons."""
    past = [g for g in know["games"] if g is not game]
    by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    decisions = []
    for g in past:
        for r in g.get("rounds", []):
            by[r["effort"]][0] += 1
            by[r["effort"]][1] += 1 if r["improved"] else 0
        decisions += [d for d in g.get("supervisor_log", []) if d.get("outcome")]
    parts = []
    if by:
        order = sorted(by, key=lambda e: EFFORTS.index(e) if e in EFFORTS else len(EFFORTS))
        parts.append("Effort levels used in earlier games (rounds played, share of rounds that improved the best "
                     "score or rank): " + ", ".join(f"{e} {by[e][0]} rounds {100 * by[e][1] // by[e][0]}%" for e in order))
    raises = [d for d in decisions if d["previous"] in EFFORTS and EFFORTS.index(d["effort"]) > EFFORTS.index(d["previous"])]
    if raises:
        parts.append(f"You raised effort {len(raises)} times; the very next round improved after "
                     f"{sum(1 for d in raises if d['outcome']['improved'])} of them.")
    solved = [len(g["guesses"]) for g in past if g.get("solved")]
    if solved:
        parts.append("Guesses needed in solved games: " + ", ".join(map(str, solved)) + ".")
    every = know.get("supervisor_lessons", [])
    feedback = [l for l in every if l.startswith(FEEDBACK_PREFIX)][-5:]
    own = [l for l in every if not l.startswith(FEEDBACK_PREFIX)][-10:]
    if feedback:
        parts.append("Feedback from the user about your effort decisions (highest priority):\n"
                     + "\n".join(f"- {l}" for l in feedback))
    if own:
        parts.append("Your own lessons from earlier games (follow them):\n" + "\n".join(f"- {l}" for l in own))
    return ("What you know from earlier games:\n" + "\n".join(parts) + "\n\n") if parts else ""


def reflect_supervisor(client, args, game: dict, know: dict) -> list[str]:
    """After a game, the supervisor reviews its own effort decisions and writes lessons about its mistakes."""
    rounds = "\n".join(f"round {r['n']}: effort {r['effort']}, best similarity {r['best_after']:.1f}, "
                       f"{'improved' if r['improved'] else 'no gain'}" for r in game.get("rounds", []))
    earlier = know.get("supervisor_lessons", [])[-10:]
    prompt = (
        f"The game is over: solved in {len(game['guesses'])} guesses over {len(game.get('rounds', []))} rounds.\n"
        f"Rounds:\n{rounds}\n\nYour effort decisions and what followed:\n"
        f"{decision_text(game.get('supervisor_log', [])) or '(none)'}\n\n"
        f"Lessons you already wrote:\n{chr(10).join('- ' + l for l in earlier) or '(none yet)'}\n\n"
        "Write 2-3 NEW one-sentence lessons about your effort decisions: moments you changed effort too late, "
        "too early or pointlessly, and what to do differently. If effort made no visible difference to the "
        "progress, say so plainly. Do not repeat a lesson you already wrote."
    )
    sup_args = argparse.Namespace(**vars(args))
    sup_args.model = args.supervisor
    return call_model(client, sup_args, SUPERVISOR_SYSTEM, prompt, LESSON_SCHEMA, "low")["lessons"]


def stall_rounds(best_per_round: list[float]) -> int:
    """Rounds since the best similarity last improved."""
    best, last = float("-inf"), 0
    for i, b in enumerate(best_per_round):
        if b > best:
            best, last = b, i
    return len(best_per_round) - 1 - last if best_per_round else 0


class Supervisor:
    """Reviews the game in a background thread while the main model thinks. A verdict is applied at the
    start of the next round, so the review never adds latency."""

    def __init__(self, args, client, emit, know: dict, game: dict):
        self.args, self.client, self.emit = args, client, emit
        self.know, self.game = know, game
        self.digest = supervisor_digest(know, game)  # fixed for the game; this game's decisions are added per review
        self.model = args.supervisor
        self.thread: threading.Thread | None = None
        self.verdict: str | None = None

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def take(self) -> str | None:
        """The effort the supervisor chose since the last call, or None."""
        verdict, self.verdict = self.verdict, None
        return verdict

    def choose_start(self, know: dict, thresholds: dict | None) -> str:
        """Picks the first round's effort. Blocks for one short call."""
        past = [len(g["guesses"]) for g in know["games"] if g.get("solved")]
        prompt = (
            f"{self.digest}A new game is starting: no guesses yet.\n"
            f"Main model: {self.args.model}.\n"
            + (f"Past solved games, guesses needed: {', '.join(map(str, past))}.\n" if past
               else "No past games yet.\n")
            + (f"Reference scores today: nearest {thresholds['nearest']}, 10th {thresholds['tenth']}, "
               f"1000th {thresholds['thousandth']}.\n" if thresholds else "")
            + "\nWhich effort level should the player use for the first round?"
        )
        sup_args = argparse.Namespace(**vars(self.args))
        sup_args.model = self.model
        try:
            data = call_model(self.client, sup_args, SUPERVISOR_SYSTEM, prompt, SUPERVISOR_SCHEMA, "low")
            new, reason = data["effort"], data["reason"]
        except Exception as e:
            new, reason = "medium", f"supervisor call failed ({type(e).__name__}); starting at medium"
        self.emit({"type": "supervisor", "effort": new, "previous": "auto", "reason": reason, "stalled": 0})
        return new

    def review(self, history: list[dict], best_per_round: list[float], effort: str,
               thresholds: dict | None) -> None:
        if self.busy():
            return
        snapshot = [dict(h) for h in history]
        self.thread = threading.Thread(
            target=self._run, args=(snapshot, list(best_per_round), effort, thresholds), daemon=True)
        self.thread.start()

    def _run(self, history, best_per_round, effort, thresholds) -> None:
        stalled = stall_rounds(best_per_round)
        ranked = [h for h in history if h["distance"] and 0 < h["distance"] < 1000]
        best_rank = max((h["distance"] for h in ranked), default=None)
        top = ", ".join("%s %.1f" % (h["guess"], h["similarity"])
                        for h in sorted(history, key=lambda h: -h["similarity"])[:8])
        mine = decision_text(list(self.game.get("supervisor_log", [])))
        prompt = (
            f"{self.digest}"
            + (f"Your decisions so far in this game:\n{mine}\n\n" if mine else "")
            + f"Rounds played: {len(best_per_round)}. Guesses: {len(history)}. Current effort: {effort}.\n"
            f"Best similarity after each round: {', '.join(f'{b:.1f}' for b in best_per_round)}.\n"
            f"Rounds since the best similarity last improved: {stalled}.\n"
            f"Best rank so far: {best_rank if best_rank is not None else 'none yet'}. "
            f"Guesses with a rank: {len(ranked)}.\n"
            + (f"Reference scores today: nearest {thresholds['nearest']}, 10th {thresholds['tenth']}, "
               f"1000th {thresholds['thousandth']}.\n" if thresholds else "")
            + f"Top guesses: {top}.\n\n"
            "Which effort level should the player use for the next round?"
        )
        sup_args = argparse.Namespace(**vars(self.args))
        sup_args.model = self.model
        try:
            data = call_model(self.client, sup_args, SUPERVISOR_SYSTEM, prompt, SUPERVISOR_SCHEMA, "low")
            new, reason = data["effort"], data["reason"]
        except Exception as e:  # a failed review falls back to a simple rule and never stops the game
            new = EFFORTS[min(EFFORTS.index(effort) + 1, len(EFFORTS) - 1)] if stalled >= 3 else effort
            reason = f"supervisor call failed ({type(e).__name__}); rule: raise one level after 3 stalled rounds"
        self.verdict = new
        self.emit({"type": "supervisor", "effort": new, "previous": effort, "reason": reason,
                   "stalled": stalled})


# ---------- game ----------

def play(args, emit, stop: threading.Event | None = None, live: dict | None = None) -> int:
    """Plays today's puzzle. Reports progress by calling emit(event_dict); returns a process exit code.

    `live` is a dict the caller may change while the game runs ("model", "supervisor"); the new values
    apply from the next round.
    Event types: start, waiting, reasoning, supervisor, model, guess, rejected, found, already_solved,
    lessons, supervisor_lessons, subagents, stopped, error.
    """
    stop = stop or threading.Event()
    client = None  # the Anthropic client is created on first use (see call_model)
    know = load_knowledge(args.knowledge)
    puzzle, thresholds = get_puzzle_info()

    # Test mode reads everything from knowledge.json but writes only to the test file, which is wiped at the
    # start of every test run. The puzzle under test is not memory, so earlier records of it are left out
    # (otherwise the solver would "remember" the secret), and a test never resumes or skips a game.
    out_path = args.knowledge
    fresh_sup: list[str] = []  # what a test run adds, kept apart so the test file holds only that
    fresh_rej: list[str] = []
    if args.test:
        out_path = args.test_file
        out_path.unlink(missing_ok=True)
        know["games"] = [g for g in know["games"] if g.get("puzzle") != puzzle]

    # The latest record of this puzzle decides: unsolved means resume it, solved means skip or replay.
    game = None if args.test else next(
        (g for g in reversed(know["games"]) if puzzle is not None and g["puzzle"] == puzzle), None)
    if game and game["solved"] and not args.replay:
        emit({"type": "already_solved", "puzzle": puzzle, "secret": game["secret"],
              "guesses": len(game["guesses"])})
        return 0
    if game and game["solved"]:
        game = None  # replay: keep the earlier game in the knowledge file and start a new record
    if not game:
        game = {"puzzle": puzzle, "date": date.today().isoformat(), "model": args.model,
                "solved": False, "secret": None, "guesses": [], "lessons": []}
        know["games"].append(game)
    game["thresholds"] = thresholds
    def save() -> None:
        save_knowledge(out_path, {"version": 1, "test_run": True, "games": [game], "supervisor_lessons": fresh_sup,
                                  "rejected": fresh_rej} if args.test else know)

    rounds = game.setdefault("rounds", [])              # one record per round: effort, progress, scout usage
    sup_log = game.setdefault("supervisor_log", [])     # every supervisor decision, with its outcome filled in later
    raw_emit = emit

    def emit(ev: dict) -> None:  # the supervisor's decisions are also kept in the game record
        if ev["type"] == "supervisor":
            sup_log.append({"after_round": len(rounds), "previous": ev["previous"], "effort": ev["effort"],
                            "reason": ev["reason"], "stalled": ev.get("stalled"), "outcome": None})
        raw_emit(ev)

    history = game["guesses"]
    resumed = len(history)
    seen = {h["guess"] for h in history} | set(know["rejected"])
    rejected: list[str] = []
    memory = build_memory(know, game)

    emit({"type": "start", "puzzle": puzzle, "model": args.model, "effort": "auto", "batch": args.batch,
          "backend": args.backend, "supervisor": args.supervisor, "test": args.test, "past_games": len(know["games"]) - 1, "thresholds": thresholds,
          "resumed": resumed})
    for i, h in enumerate(history, 1):
        emit({"type": "guess", "n": i, "word": h["guess"], "similarity": h["similarity"],
              "distance": h["distance"], "resumed": True})

    limit = args.max_guesses or float("inf")
    stalls = 0
    effort = "auto"  # effort is never set by hand: the supervisor picks it (medium when the supervisor is off)
    best_per_round: list[float] = [r["best_after"] for r in rounds]
    supervisor: Supervisor | None = None
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
            # Haiku has no effort setting, so there is nothing for a supervisor to adjust.
            if args.supervisor == "off" or not effort_ok(args.model):
                supervisor = None
            elif supervisor is None:
                supervisor = Supervisor(args, client, emit, know, game)
            if effort == "auto":  # the supervisor picks the starting level; without one, medium
                effort = supervisor.choose_start(know, thresholds) if supervisor else "medium"
            if supervisor:
                effort = supervisor.take() or effort  # verdict from the review that ran during the last round
                if history:
                    supervisor.review(history, best_per_round, effort, thresholds)
            emit({"type": "waiting", "effort": effort})
            scouts = run_scouts(client, args, history, rejected, thresholds, emit)  # parallel, on a cheaper model
            prev_best = max((h["similarity"] for h in history), default=None)
            prev_rank = max((h["distance"] for h in history if h["distance"] and h["distance"] > 0), default=0)
            accepted: list[dict] = []
            data = call_model(client, args, SYSTEM,
                               build_prompt(memory, history, rejected, args.batch, thresholds, scouts),
                               SCHEMA, effort)
            emit({"type": "reasoning", "text": data["reasoning"], "guesses": data["guesses"],
                  "effort": effort, "thinking_tokens": (data.get("_meta") or {}).get("thinking_tokens")})
            progressed = False
            for word in data["guesses"]:
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
                scout_use = {role: {"proposed": len(words),
                                    "used": [[h["guess"], h["similarity"], h["distance"]]
                                             for h in accepted if h["guess"] in words]}
                             for role, words in scouts.items()}
                rounds.append({"n": len(rounds) + 1, "model": args.model, "effort": effort, "improved": improved,
                               "best_after": best_after, "rank_after": rank_after,
                               "words": [h["guess"] for h in accepted],
                               "thinking_tokens": (data.get("_meta") or {}).get("thinking_tokens"),
                               "scouts": scout_use})
                for d in sup_log:  # the first decision at this effort that has not been judged yet gets its outcome
                    if d["outcome"] is None and d["effort"] == effort and d["after_round"] < rounds[-1]["n"]:
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
        try:
            game["lessons"] = write_lessons(client, args, game, know)
            emit({"type": "lessons", "lessons": game["lessons"]})
        except Exception as e:  # the game is won; never lose it over a failed reflection call
            emit({"type": "error", "message": f"could not write lessons: {e}"})
        if sup_log and args.supervisor != "off":
            try:  # the supervisor reviews its own effort decisions and keeps lessons about its mistakes
                new = reflect_supervisor(client, args, game, know)
                know["supervisor_lessons"] = (know.get("supervisor_lessons", []) + new)[-30:]
                fresh_sup.extend(new)
                emit({"type": "supervisor_lessons", "lessons": new})
            except Exception as e:
                emit({"type": "error", "message": f"supervisor could not write lessons: {e}"})
        save()
        return 0

    save()
    reason = ("stopped by user" if stop.is_set()
              else "no new words proposed" if stalls >= STALL_LIMIT else "guess limit reached")
    emit({"type": "stopped", "reason": reason, "best": best_guess(history)})
    return 1


def add_user_feedback(args, rating: str, text: str) -> bool:
    """Stores the user's feedback on the supervisor, where the next game's supervisor will read it first."""
    label = {"low": "the effort was too low", "ok": "the effort level was fine",
             "high": "the effort was too high"}.get(rating, "")
    line = f"{FEEDBACK_PREFIX}: " + "; ".join(x for x in (label, text.strip()) if x)
    if line == f"{FEEDBACK_PREFIX}: ":
        return False
    path = args.test_file if args.test else args.knowledge
    know = load_knowledge(path)
    know["supervisor_lessons"].append(line)
    know["supervisor_lessons"] = know["supervisor_lessons"][-60:]
    if know["games"]:
        know["games"][-1]["supervisor_feedback"] = {"rating": rating, "text": text.strip()}
    save_knowledge(path, know)
    return True


def best_guess(history: list[dict]) -> str | None:
    return max(history, key=lambda h: h["similarity"])["guess"] if history else None


# ---------- command line ----------

def make_cli_emit(verbose: bool):
    def emit(ev: dict) -> None:
        t = ev["type"]
        if t == "start":
            resumed = f" | resuming {ev['resumed']} saved guesses" if ev["resumed"] else ""
            print(f"Puzzle {ev['puzzle']} | model: {ev['model']} | effort: {ev['effort']} | "
                  f"batch: {ev['batch']} | past games: {ev['past_games']}{resumed}")
        elif t == "reasoning" and verbose:
            print(f"  [claude/{ev['effort']}] {ev['text']}")
        elif t == "supervisor":
            change = f"{ev['previous']} -> {ev['effort']}" if ev["effort"] != ev["previous"] else f"keep {ev['effort']}"
            print(f"  [supervisor] {change}: {ev['reason']}")
        elif t == "guess" and not ev.get("resumed"):
            rank = ev["distance"]
            tag = f"{rank}/1000" if rank and rank > 0 else "far"
            print(f"#{ev['n']:>3} {ev['word']}\t{ev['similarity']:6.2f}\t{tag}")
        elif t == "rejected":
            print(f"  {ev['word']}: not in vocabulary")
        elif t == "found":
            print(f"\nFound the secret word: {ev['secret']} in {ev['guesses']} guesses")
        elif t == "already_solved":
            print(f"Puzzle {ev['puzzle']} already solved: {ev['secret']} ({ev['guesses']} guesses). "
                  "Use --replay to play it again.")
        elif t == "lessons":
            for lesson in ev["lessons"]:
                print(f"  lesson: {lesson}")
        elif t == "supervisor_lessons":
            for lesson in ev["lessons"]:
                print(f"  supervisor lesson: {lesson}")
        elif t == "subagents":
            for sc in ev["scouts"]:
                print(f"  [{sc['role']}] {', '.join(sc['candidates']) or '(nothing)'}")
        elif t == "stopped":
            print(f"\nStopped ({ev['reason']}). Best: {ev['best'] or '-'}. Progress saved; run again to resume.")
        elif t == "error":
            print(f"\nClaude call failed: {ev['message']}\nProgress saved; run again to resume.")
    return emit


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="sonnet",
                   help=f"alias ({', '.join(MODELS)}: always the newest of that family) or any full model ID, "
                        "e.g. claude-sonnet-5 (default: sonnet)")
    p.add_argument("--supervisor", default="haiku",
                   help="model that watches the game in the background and adjusts the solver's effort "
                        "(alias or model ID; 'off' disables; no effect when the solver itself is Haiku)")
    p.add_argument("--subagents", type=int, default=3, choices=[0, 1, 2, 3],
                   help="scouts that explore in parallel each round and feed the solver proposals "
                        "(field-scout, neighbour-scout, triangulator); 0 disables (default 3)")
    p.add_argument("--subagent-model", default="haiku",
                   help="model the scouts run on (alias, model ID or provider:model; default haiku)")
    p.add_argument("--backend", default="auto", choices=["auto", "cli", "api"],
                   help="how to reach each provider. cli: the provider's own CLI signed in with your "
                        "subscription (Claude Code, Codex, Gemini CLI), no key. api: an API key. "
                        "auto (default): the CLI when installed, otherwise the key. Local servers "
                        "(Ollama, LM Studio) ignore this")
    p.add_argument("--claude-bin", help="path to the Claude Code binary (cli backend; auto-detected)")
    p.add_argument("--batch", type=int, default=5, help="guesses per Claude call (default 5)")
    p.add_argument("--max-guesses", type=int, default=0,
                   help="stop after this many accepted guesses (default 0 = no limit)")
    p.add_argument("--knowledge", type=Path, default=KNOWLEDGE_PATH, help="path of the knowledge JSON")
    p.add_argument("--test", action="store_true",
                   help="test run: read memory from --knowledge but write only to --test-file, which is wiped at "
                        "the start of every test run; --knowledge is never modified")
    p.add_argument("--test-file", type=Path, default=Path(__file__).with_name("knowledge_test.json"),
                   help="where a --test run writes (default knowledge_test.json)")
    p.add_argument("--replay", action="store_true",
                   help="play again even if today's puzzle is already solved (the earlier game is kept; "
                        "the web UI always replays)")
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
                            "feedback": add_user_feedback,
                            "load_knowledge": lambda: knowledge_summary(load_knowledge(args.knowledge))})
    return play(args, cli_emit)


if __name__ == "__main__":
    sys.exit(main())
