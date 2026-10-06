"""Models other than Claude.

A model is named "provider:model", e.g. "openai:gpt-5", "google:gemini-2.5-pro" or "ollama:qwen3:8b".
Every provider can be reached in up to two ways, like Claude:

- api: an HTTP API key in an environment variable. All providers here speak the OpenAI-compatible chat
  API, so one client covers them all (Google exposes such an endpoint for Gemini).
- cli: the provider's official command-line tool, signed in with your subscription or account instead of
  a key (Codex CLI for OpenAI, Gemini CLI for Google).

Local servers (Ollama, LM Studio) need neither. Add any other OpenAI-compatible server through the
"custom" provider (OPENAI_COMPAT_BASE_URL, optional OPENAI_COMPAT_API_KEY) or by adding a line to PROVIDERS.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROVIDERS = {
    "openai": {"label": "OpenAI", "base": "https://api.openai.com/v1", "key_env": ["OPENAI_API_KEY"],
               "cli": "codex"},
    "google": {"label": "Google Gemini", "base": "https://generativelanguage.googleapis.com/v1beta/openai",
               "key_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "cli": "gemini"},
    "ollama": {"label": "Ollama (local)", "base": "http://localhost:11434/v1", "base_env": "OLLAMA_HOST",
               "local": True},
    "lmstudio": {"label": "LM Studio (local)", "base": "http://localhost:1234/v1", "local": True},
    "openrouter": {"label": "OpenRouter", "base": "https://openrouter.ai/api/v1",
                   "key_env": ["OPENROUTER_API_KEY"]},
    "groq": {"label": "Groq", "base": "https://api.groq.com/openai/v1", "key_env": ["GROQ_API_KEY"]},
    "mistral": {"label": "Mistral", "base": "https://api.mistral.ai/v1", "key_env": ["MISTRAL_API_KEY"]},
    "deepseek": {"label": "DeepSeek", "base": "https://api.deepseek.com/v1", "key_env": ["DEEPSEEK_API_KEY"]},
    "xai": {"label": "xAI", "base": "https://api.x.ai/v1", "key_env": ["XAI_API_KEY"]},
    "together": {"label": "Together", "base": "https://api.together.xyz/v1", "key_env": ["TOGETHER_API_KEY"]},
    "custom": {"label": "Custom OpenAI-compatible", "base_env": "OPENAI_COMPAT_BASE_URL",
               "key_env": ["OPENAI_COMPAT_API_KEY"]},
}

REASONING_MODEL = re.compile(r"^(o\d|gpt-5)", re.I)  # OpenAI models that accept a reasoning effort
HTTP_TIMEOUT = 600


ENV_PATH = Path(__file__).with_name(".env")  # where keys entered in the page are kept; ignored by git

# What can be entered in the page. A slot is one environment variable: the keys, and the address of a custom server.
KEY_SLOTS = (
    [{"id": "anthropic", "label": "Anthropic (Claude API)", "env": "ANTHROPIC_API_KEY", "secret": True}]
    + [{"id": name, "label": cfg["label"], "env": cfg["key_env"][0], "secret": True}
       for name, cfg in PROVIDERS.items() if cfg.get("key_env") and name != "custom"]
    + [{"id": "custom", "label": "Custom OpenAI-compatible server: key", "env": "OPENAI_COMPAT_API_KEY", "secret": True},
       {"id": "custom_base", "label": "Custom OpenAI-compatible server: address", "env": "OPENAI_COMPAT_BASE_URL",
        "secret": False}]
)
_SECRET_OK = re.compile(r"^[A-Za-z0-9._\-:/@+=~]{8,400}$")
_URL_OK = re.compile(r"^https?://[^\s\"']{3,300}$")


def split_model(model: str) -> tuple[str | None, str]:
    """('openai', 'gpt-5') for 'openai:gpt-5'; (None, model) for Claude names, which have no provider prefix."""
    prefix, _, rest = model.partition(":")
    if rest and prefix in PROVIDERS:
        return prefix, rest
    return None, model


def effort_supported(provider: str, name: str) -> bool:
    """Only OpenAI's reasoning models take an effort level, so only they can be supervised."""
    return provider == "openai" and bool(REASONING_MODEL.match(name))


def base_url(cfg: dict) -> str | None:
    env = os.environ.get(cfg.get("base_env", ""))
    url = env or cfg.get("base")
    if not url:
        return None
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")
    if env and cfg.get("base_env") == "OLLAMA_HOST" and not url.endswith("/v1"):
        url += "/v1"  # OLLAMA_HOST is a bare host:port; the compatible API lives under /v1
    return url


def api_key(cfg: dict) -> str | None:
    return next((os.environ[e] for e in cfg.get("key_env", []) if os.environ.get(e)), None)


def cli_path(cfg: dict) -> str | None:
    return shutil.which(cfg["cli"]) if cfg.get("cli") else None


def _local_up(cfg: dict) -> bool:
    try:
        return requests.get(f"{base_url(cfg)}/models", timeout=1.5).status_code == 200
    except requests.RequestException:
        return False


def provider_status() -> dict:
    """Per provider: is an API key set, is the official CLI installed, is a local server running."""
    out = {}
    for name, cfg in PROVIDERS.items():
        out[name] = {"label": cfg["label"], "key": bool(api_key(cfg)), "key_env": cfg.get("key_env", [None])[0],
                     "cli": cli_path(cfg) is not None, "cli_name": cfg.get("cli"), "local": bool(cfg.get("local")),
                     "up": _local_up(cfg) if cfg.get("local") else None}
    return out


def pick_mode(name: str, cfg: dict, backend: str) -> str:
    """'http' for local servers and API keys, 'cli' for the official CLI. backend: auto, cli or api."""
    if cfg.get("local"):
        return "http"
    has_cli, has_key = cli_path(cfg) is not None, api_key(cfg) is not None
    if backend == "cli" and has_cli:
        return "cli"
    if backend == "api" and (has_key or not cfg.get("key_env")):
        return "http"
    if backend == "auto":
        if has_cli:
            return "cli"
        if has_key or not cfg.get("key_env"):
            return "http"
    options = []
    if cfg.get("key_env"):
        options.append(f"set {cfg['key_env'][0]} (API key)")
    if cfg.get("cli"):
        options.append(f"install and sign in to the `{cfg['cli']}` CLI (no key needed)")
    raise RuntimeError(f"{cfg['label']} is not connected for backend '{backend}'. Either " + " or ".join(options) + ".")


# ---------- calling ----------

def extract_json(text: str) -> dict:
    """The JSON object in a model reply, tolerating <think> blocks, code fences and chatter around it."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    for candidate in (text, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise RuntimeError(f"model did not return JSON: {text[:300]!r}")


def _checked(data: dict, schema: dict, who: str) -> dict:
    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        raise RuntimeError(f"{who} reply is missing {missing}: {json.dumps(data, ensure_ascii=False)[:300]}")
    return data


def _shape(schema: dict) -> str:
    return "\n\nReply with only one JSON object of this shape, no other text:\n" + json.dumps(schema)


def _effort_value(effort: str) -> str:
    return {"xhigh": "high", "max": "high"}.get(effort, effort)


TOO_LARGE = re.compile(r"request too large|reduce max_tokens", re.I)   # one reply bigger than the per-minute output limit
RETRY_IN = re.compile(r"try again in (?:(\d+)m)?\s*([\d.]+)(ms|s)", re.I)
RATE_RETRIES, RATE_MAX_WAIT = 4, 60


def _post(url: str, headers: dict, payload: dict, label: str):
    """POST; a per-minute rate limit (HTTP 429 with a wait time) is waited out, up to RATE_RETRIES times."""
    for attempt in range(RATE_RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
        except requests.RequestException as e:
            raise RuntimeError(f"{label} unreachable: {e}")
        m = RETRY_IN.search(r.text) if r.status_code == 429 and not TOO_LARGE.search(r.text) else None
        if not m or attempt == RATE_RETRIES:
            return r
        wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) / (1000 if m.group(3) == "ms" else 1))
        time.sleep(min(wait + 0.5, RATE_MAX_WAIT))
    return r


def call_http(provider: str, name: str, cfg: dict, system: str, prompt: str, schema: dict, effort: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if key := api_key(cfg):
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": name, "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": prompt + _shape(schema)}]}
    if effort_supported(provider, name):
        body["reasoning_effort"] = _effort_value(effort)
    formats = [{"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": True}},
               {"type": "json_object"}, None]  # strictest first; servers that reject a format get the next one
    last = ""
    for fmt in formats:
        payload = {**body, **({"response_format": fmt} if fmt else {})}
        url = f"{base_url(cfg)}/chat/completions"
        r = _post(url, headers, payload, cfg["label"])
        if r.status_code in (413, 429) and "max_tokens" not in body and TOO_LARGE.search(r.text):
            # the account's output-tokens-per-minute limit is below the model's default reply size: cap the reply
            m = re.search(r"Limit (\d+)", r.text)
            body["max_tokens"] = max(200, int(int(m.group(1)) * 0.8)) if m else 800
            payload = {**body, **({"response_format": fmt} if fmt else {})}
            r = _post(url, headers, payload, cfg["label"])
        if r.status_code in (400, 422) and fmt is not None:
            last = r.text[:300]
            continue
        if r.status_code != 200:
            raise RuntimeError(f"{cfg['label']} error {r.status_code}: {r.text[:300]}")
        reply = r.json()
        text = reply["choices"][0]["message"].get("content") or ""
        data = _checked(extract_json(text), schema, cfg["label"])
        details = (reply.get("usage") or {}).get("completion_tokens_details") or {}
        return {**data, "_meta": {"thinking_tokens": details.get("reasoning_tokens")}}
    raise RuntimeError(f"{cfg['label']} rejected every response format: {last}")


def call_cli(provider: str, name: str, cfg: dict, system: str, prompt: str, schema: dict, effort: str) -> dict:
    """OpenAI through the Codex CLI or Google through the Gemini CLI, signed in with an account."""
    exe = cli_path(cfg)
    full = f"{system}\n\n---\n\n{prompt}{_shape(schema)}"
    model_args = [] if name in ("", "default") else ["-m", name]
    with tempfile.TemporaryDirectory() as tmp:
        if provider == "openai":
            out_file = str(Path(tmp) / "reply.txt")
            cmd = [exe, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-o", out_file, *model_args]
            if effort_supported(provider, name):
                cmd += ["-c", f'model_reasoning_effort="{_effort_value(effort)}"']
            cmd.append("-")  # the prompt comes from stdin
            stdin = full
        else:
            out_file = None
            cmd = [exe, *model_args, "-p", "Follow the instructions on stdin and reply with only the JSON object."]
            stdin = full
        proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True, encoding="utf-8", cwd=tmp,
                              timeout=HTTP_TIMEOUT)
        text = Path(out_file).read_text(encoding="utf-8") if out_file and Path(out_file).exists() else proc.stdout
    if proc.returncode != 0 and not text.strip():
        raise RuntimeError(f"{cfg['cli']} CLI failed (exit {proc.returncode}): {(proc.stderr or proc.stdout)[:400]}")
    return {**_checked(extract_json(text), schema, cfg["label"]), "_meta": {"thinking_tokens": None}}


def call(args, provider: str, name: str, system: str, prompt: str, schema: dict, effort: str) -> dict:
    cfg = PROVIDERS[provider]
    if pick_mode(provider, cfg, getattr(args, "backend", "auto")) == "cli":
        return call_cli(provider, name, cfg, system, prompt, schema, effort)
    return call_http(provider, name, cfg, system, prompt, schema, effort)


# ---------- keys entered in the page ----------

def _slot(slot_id: str) -> dict:
    for slot in KEY_SLOTS:
        if slot["id"] == slot_id:
            return slot
    raise ValueError(f"unknown key slot: {slot_id}")


def key_status() -> list[dict]:
    """Which keys are set. A secret is never returned, only whether it is set and its last four characters."""
    out = []
    for slot in KEY_SLOTS:
        value = os.environ.get(slot["env"], "")
        out.append({"id": slot["id"], "label": slot["label"], "env": slot["env"], "secret": slot["secret"],
                    "set": bool(value),
                    "hint": (value[-4:] if slot["secret"] and len(value) >= 12 else "") if slot["secret"] else value})
    return out


def _write_env(name: str, value: str | None) -> None:
    """Sets, replaces or removes one NAME=value line of .env, keeping every other line and comment as they are."""
    lines = ENV_PATH.read_text(encoding="utf-8-sig").splitlines() if ENV_PATH.exists() else []
    kept, found = [], False
    for line in lines:
        bare = line.strip().removeprefix("export ").strip()
        if bare.split("=", 1)[0].strip() == name and "=" in bare:
            found = True
            if value is not None:
                kept.append(f"{name}={value}")
        else:
            kept.append(line)
    if not found and value is not None:
        kept.append(f"{name}={value}")
    tmp = ENV_PATH.with_name(f"{ENV_PATH.name}.{os.getpid()}.tmp")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
    tmp.replace(ENV_PATH)


def save_key(slot_id: str, value: str, persist: bool = True) -> None:
    """Uses the key from now on (this process), and keeps it in .env when `persist`. An empty value removes it.
    Raises ValueError for anything that does not look like a key or an address, so a line break or a quote can
    never end up in .env."""
    slot = _slot(slot_id)
    value = (value or "").strip()
    if value:
        ok = _SECRET_OK if slot["secret"] else _URL_OK
        if not ok.match(value):
            raise ValueError("that does not look like a " + ("key" if slot["secret"] else "web address (http://...)"))
        os.environ[slot["env"]] = value
    else:
        os.environ.pop(slot["env"], None)
    if persist:
        _write_env(slot["env"], value or None)
    reset_cache()


def reset_cache() -> None:
    global _cache
    _cache = (0.0, [])


def check_key(slot_id: str) -> dict:
    """Asks the provider whether the key works, with the cheapest call it has: listing its models."""
    slot = _slot(slot_id)
    value = os.environ.get(slot["env"], "")
    if not value:
        return {"ok": False, "message": "not set"}
    try:
        if slot_id == "anthropic":
            r = requests.get("https://api.anthropic.com/v1/models", params={"limit": 1}, timeout=15,
                             headers={"x-api-key": value, "anthropic-version": "2023-06-01"})
        elif slot_id == "custom_base":
            r = requests.get(f"{value.rstrip('/')}/models", timeout=8)
        else:
            cfg = PROVIDERS["custom" if slot_id == "custom" else slot_id]
            base = base_url(cfg)
            if not base:
                return {"ok": False, "message": "set the server address first"}
            r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {value}"}, timeout=15)
    except requests.RequestException as e:
        return {"ok": False, "message": f"could not reach the server ({type(e).__name__})"}
    if r.status_code == 200:
        return {"ok": True, "message": "works"}
    if r.status_code in (401, 403):
        return {"ok": False, "message": f"the provider rejected the key (HTTP {r.status_code})"}
    return {"ok": False, "message": f"unexpected answer (HTTP {r.status_code})"}


# ---------- listing models for the UI ----------

_cache: tuple[float, list[dict]] = (0.0, [])


# Models that cannot play a word game (they do not chat): embeddings, speech, images, moderation, rerankers.
NOT_CHAT = re.compile(r"embed|whisper|tts|dall-e|moderation|rerank|davinci|babbage|transcribe|sora|"
                      r"realtime|audio|image|vision-only|guard", re.I)


def _list_one(provider: str, cfg: dict) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key(cfg)}"} if api_key(cfg) else {}
    try:
        r = requests.get(f"{base_url(cfg)}/models", headers=headers, timeout=4)
        if r.status_code != 200:
            return []
        ids = [m.get("id") or m.get("name") for m in r.json().get("data", [])]
    except (requests.RequestException, ValueError):
        return []
    # Google lists ids as "models/gemini-..."; the chat endpoint wants the bare name.
    return [{"id": f"{provider}:{i.removeprefix('models/')}", "label": i.removeprefix("models/"),
             "group": cfg["label"]} for i in sorted(ids) if i and not NOT_CHAT.search(i)]


def local_models(limit: int = 4) -> list[dict]:
    """Models on local servers (Ollama, LM Studio) that are running right now; empty when none is."""
    out: list[dict] = []
    for name, cfg in PROVIDERS.items():
        if cfg.get("local") and _local_up(cfg):
            out += _list_one(name, cfg)[:limit]
    return out


def list_provider_models() -> list[dict]:
    """Models from every provider that is reachable: has a key, or is a local server that is running.
    A CLI-only provider (no key) gets one 'CLI default' entry, since a CLI cannot list models."""
    global _cache
    if time.time() - _cache[0] < 120 and _cache[1]:
        return _cache[1]
    jobs = {p: c for p, c in PROVIDERS.items() if (c.get("local") or api_key(c) or (c.get("base_env") and base_url(c)))}
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        for models in pool.map(lambda kv: _list_one(*kv), jobs.items()):
            out.extend(models)
    for provider, cfg in PROVIDERS.items():
        if cli_path(cfg) and not api_key(cfg):
            out.append({"id": f"{provider}:default", "label": f"CLI default model", "group": cfg["label"]})
    _cache = (time.time(), out)
    return out
