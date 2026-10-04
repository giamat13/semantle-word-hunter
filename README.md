# Semantle Word Hunter

Plays [Hebrew Semantle](https://semantle.ishefi.com) (סמנטעל) for you. An AI model (Claude, OpenAI, Google,
Ollama and more) picks the guesses, and the hunter **learns from every game it plays**.

Semantle is a word-guessing game: a secret word is chosen each day, and every guess returns a
similarity score from a word-embedding model. This solver asks the model for guesses, submits them,
feeds the scores back, and repeats until the secret word is found.

> Unofficial project. Not affiliated with the Semantle site or its author.

## How it works

1. The model sees every guess so far (word, similarity, rank) plus a digest of earlier games and proposes new
   Hebrew words. **It chooses how many** (1 to 10) each round: wide batches while mapping the territory, one
   to three once a field is known, so that every result steers the next guess. There is no batch-size setting.
2. The script submits them to the site's public `/api/distance` endpoint.
3. The loop ends when the site reports rank 1000, the secret word.
4. When the game ends, the model writes a **summary** of it and a few short lessons. The summary and the
   lessons are saved to `knowledge.json`; the raw guesses are not.

### Learning across days

`knowledge.json` stores, for every game, its counts (guesses, rounds, best score and rank), the setup it ran
with, the secret word, a **summary** the model wrote when the game ended, and its lessons. It does **not** keep
the guesses themselves: whatever was worth keeping from them goes into the summary, which covers the field, the
words that ranked closest with their ranks, the turning point and what was wasted. Each new game starts with a
digest built from it:

- past secret words (never the answer again, but evidence about the embedding),
- the summaries of the most recent games,
- how each scout's words fared,
- recent lessons.

The game is saved as it goes. Running again always starts a **new** game: whatever happened earlier
today is ignored (see below).
Similarity scores depend on the day's secret word, so what carries over is *search strategy* and
knowledge about the embedding, not answers. The digest becomes more useful after several days of play.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Pick one backend:

| Backend | Flag | Uses | Needs |
|---|---|---|---|
| Claude Code (default) | `--backend cli` | your Claude subscription's usage limit | [Claude Code](https://claude.com/claude-code) installed and logged in |
| Anthropic API | `--backend api` | API credit | `ANTHROPIC_API_KEY` (or `ant auth login`) |

With the `cli` backend each call runs `claude -p` with tools, MCP servers, settings and slash commands
disabled, which keeps a call at roughly 3K tokens. The binary is auto-detected (PATH, or the VS Code
extension's bundled copy); override with `--claude-bin`.

### Other providers: OpenAI, Google, Ollama and more

Name a model as `provider:model`, for example `openai:gpt-5`, `google:gemini-2.5-pro`, `ollama:qwen3:8b`.
No prefix means Claude. Like Claude, every cloud provider can be reached two ways:

| Provider | API key | Subscription / account, no key |
|---|---|---|
| Claude | `ANTHROPIC_API_KEY` | Claude Code (`claude`) |
| OpenAI | `OPENAI_API_KEY` | Codex CLI (`codex`), signed in with ChatGPT |
| Google | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Gemini CLI (`gemini`), signed in with Google |
| OpenRouter, Groq, Mistral, DeepSeek, xAI, Together | `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `TOGETHER_API_KEY` | none |
| Ollama, LM Studio | none (local server; `OLLAMA_HOST` to change the address) | none |
| Anything else OpenAI-compatible | `OPENAI_COMPAT_BASE_URL` (+ `OPENAI_COMPAT_API_KEY`), provider `custom` | none |

`--backend auto` (default) uses the provider's CLI when it is installed and falls back to the key;
`--backend cli` or `--backend api` forces one. Models are listed from each provider's own `/models`
endpoint (and Claude's Models API), so a new model appears without any code change. Type any ID that is
not listed. To add a provider, add a line to `PROVIDERS` in [providers.py](providers.py).

Notes: the supervisor only adjusts effort on models that have it (Claude except Haiku, and OpenAI
reasoning models). Small local models can struggle with the long prompt and the JSON format. Ollama's
default context window is small, raise it with `OLLAMA_CONTEXT_LENGTH` if answers look cut off.
The Codex and Gemini CLI paths follow those tools' documented flags but were not run in development.

## Usage

```bash
python solver.py                       # today's puzzle, Sonnet, no guess limit
python solver.py --model haiku -v      # cheaper model, print Claude's reasoning
python solver.py --model opus --supervisor sonnet
python solver.py --max-guesses 40      # cap the number of accepted guesses
python solver.py --backend api         # use the Anthropic API instead
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | `sonnet` | `opus`, `sonnet`, `haiku`, `fable`, or any full model ID |
| `--supervisor` | `haiku` | model that watches in the background and adjusts effort; `off` disables |
| `--test` / `--test-file` | off / `knowledge_test.json` | read from `--knowledge`, write only to the test file (wiped each test run) |
| `--subagents` | `auto` | scouts per round: `auto` lets the supervisor decide (starting at 3), a number `0` to `3` is fixed for the whole game |
| `--subagent-model` | `haiku` | model the scouts run on |
| `--ui` / `--autostart` | off | serve the web UI / start solving immediately |
| `--port` | `8765` | first port to try for the UI |
| `--max-guesses` | `0` (none) | stop after this many accepted guesses |
| `--knowledge` | `knowledge.json` | path of the learning file |
| `-v` | off | print Claude's reasoning each round |

If today's puzzle is already solved, the script says so and exits.

### Web UI

```bash
python solver.py --ui --autostart      # opens http://127.0.0.1:8765 and starts solving
python solver.py --ui                  # opens the UI and waits for you to press start
```

The page shows, live: the closest word so far on a thermometer scaled with today's real reference scores,
the search phase (explore, locate, converge), the trail of every guess as a chart and a sortable table,
Claude's reasoning round by round with the effort level used, supervisor decisions, and what was learned
in earlier games. You can pick the model and supervisor, start and stop runs, and switch
light/dark (`?theme=dark` forces a theme for one tab).
The server is stdlib-only, listens on 127.0.0.1 and rejects POSTs from other origins.

### Daily run from VS Code

[.vscode/launch.json](.vscode/launch.json) defines two launch configurations. Opening the folder and
pressing **F5** starts the web UI and solves today's puzzle with Sonnet; the second configuration runs
in the terminal only.

### Race mode

The settings panel has a race editor: two to four lanes, each with its own model, number of scouts and
supervisor. **Start race** makes every lane hunt today's puzzle at the same time, so setups are compared on
exactly the same puzzle, in one day instead of over weeks. Each lane runs in test mode (memory is read from
`knowledge.json`, nothing is written to it) and has a live thermometer, its guess count, calls used and time.
When all lanes are done, a table shows who solved it, with fewest guesses and fastest. The result is kept in
`knowledge_races.json` (`--races-file`), and the statistics panel aggregates races per setup.
Every lane spends its own usage, so a race of three costs about three games; `max guesses` caps each lane.
Pressing run or race while something is running stops it first.

### Statistics and sharing

The bottom of the page shows how games went across days: a line of guesses per game, and one row per setup
(model, number of scouts, supervisor on or off) with every game as a dot and the median as a vertical line.
It exists to answer whether the scouts, the supervisor or a model change really help; that needs a few games
per setup, and the page says so while there are fewer than three. Replays of an already solved puzzle are not
counted, because the solver has already seen the answer.

After a win, **Copy for sharing** puts a spoiler-free Wordle-style summary on the clipboard: puzzle number,
guesses, rounds, the setup, and one coloured square per guess (white far, yellow warm, orange in the top 1000,
green found). It never contains a guessed word.

### Every run starts fresh

However a run is started (terminal, F5, the Run button, a race, test mode), it ignores everything that
happened earlier today. It never resumes an unfinished game, never skips a puzzle that is already solved,
and it leaves today's earlier records out of what the solver and the supervisor read: the summaries,
past secrets, lessons, effort statistics, and the supervisor lessons those games wrote. Otherwise the solver
would "remember" the secret. The records stay in `knowledge.json` and the statistics panel still counts them;
from tomorrow on they are ordinary history and are learned from.

### Test runs

`--test` makes a run that **reads** its memory from `knowledge.json` and **writes only** to
`knowledge_test.json`, which is wiped at the start of every test run. `knowledge.json` is never modified by
a test. The puzzle being tested is left out of the memory the solver reads, so it cannot "remember" that
day's secret. In VS Code, the "TEST run" launch configuration (F5)
does this. The web UI also has a test-mode checkbox in the settings.

### Sub-agents

Each round, up to three scouts explore in parallel on a cheaper model (`--subagent-model`, default Haiku)
and hand the solver their proposals, which its system prompt tells it to use:

- **field-scout**: fields that have not been probed yet,
- **neighbour-scout**: close relatives of the best words,
- **triangulator**: what the best words share that the weak ones lack, including the outcome, the opposite and
  the cause. It exists for the common stall where the solver circles one cluster of near-synonyms.

The solver still decides the final batch. Which scout's words were actually used and how they scored is
recorded, and that track record goes back into the prompt. `--subagents auto` (the default) lets the supervisor
choose the number each round; `--subagents 0` turns them off and `1` to `3` fix the number, so the supervisor
cannot change it. Use a fixed number when you want a clean comparison, for example in a race. The scouts add calls to every round, so they cost usage; they run in parallel, so they add
little time.

### Supervisor and thinking

- A background **supervisor** (`--supervisor haiku` by default, `off` to disable) reviews the game while the
  main model is thinking: best score per round, rounds without improvement, ranks. It may raise the main
  model's effort when the search stalls, or lower it when progress is easy. A verdict applies from the
  next round, so it never adds latency. It has no effect when the solver itself is Haiku, which has no
  effort setting.
- The supervisor also manages the **scout budget** each round: how many scouts (0 to 3) and which tier, `cheap`
  (the scout model) or `strong` (the same model as the player). It saves when the search is moving and adds
  scouts, then the strong tier, when it stalls. With a model that has no effort levels (Haiku, most local
  models) only the scout budget is adjusted.
- The supervisor **learns from the record**. Every round is saved in `knowledge.json` (effort used, whether the
  best score or rank improved) together with each supervisor decision and what the next round showed. Its
  prompt includes how often each effort level led to progress, how its earlier raises turned out, and lessons
  it wrote itself after earlier games about its own mistakes (raising too late, too early, or pointlessly).
- **Extended thinking is always on**: adaptive thinking for Opus, Sonnet and Fable, and a fixed budget for
  Haiku. Adaptive models decide how much to think; effort controls how deep they go, which is what the
  supervisor adjusts.

## Be kind to the site

The solver uses the site's public API, a few requests per round. Please run it for your own daily puzzle,
not in bulk, and don't hammer the endpoint. The API is undocumented and may change; if it does, the
endpoint and header live at the top of `solver.py` (`SITE`, `HEADERS`).

## Files

- `solver.py`: the solver (prompt, site client, Claude backends, supervisor, knowledge handling)
- `ui_server.py`, `ui.html`: the web UI (local server and single-page front end)
- `knowledge.json`: created on first run; your game history and lessons
- `requirements.txt`, `.vscode/launch.json`

## License

[MIT](LICENSE)

---

## בקצרה (עברית)

פותר אוטומטי לסמנטעל בעברית. Claude מציע ניחושים, הסקריפט שולח אותם לאתר, וכשהמילה נמצאת Claude כותב לקחים.
הכול נשמר ב-`knowledge.json` ומשמש את המשחק הבא. ברירת המחדל משתמשת במכסת המנוי של Claude Code
(`--backend cli`), ואפשר גם API עם `--backend api`. הרצה יומית: F5 ב-VS Code, או `python solver.py`.
