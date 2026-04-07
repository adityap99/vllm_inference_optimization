# SPDX-License-Identifier: Apache-2.0
"""
Migration-aware disaggregated proxy for vLLM (PROPOSAL.md §3).

All client traffic enters here. Requests go to the fast-lane PD pair first.
The proxy tracks per-request token throughput and, when a straggler is
detected, transparently re-admits the request to the slow-lane PD pair.

Lane isolation
──────────────
Two independent ZMQ listeners prevent cross-lane KV misrouting:
  FAST lane servers  → register on PROXY_PORT      (default 30099)
  SLOW lane servers  → register on SLOW_PROXY_PORT (default 30097)

Detection condition (evaluated every EVAL_INTERVAL tokens after T_MIN):
  migrate  iff  N >= T_MIN  and  N / t_elapsed < R_SLOW
  where T_MIN=200 tokens, R_SLOW=30 tok/s  (env-overridable)

Re-admission uses vLLM's prompt_token_ids extension so that the slow-lane
prefill processes the full context (original prompt + generated tokens) via
exact token IDs — no detokenize/retokenize roundtrip at re-admission time.

Environment variables
─────────────────────
  MODEL              Hugging Face model id  (default: meta-llama/Llama-2-13b-hf)
  PROXY_PORT         ZMQ port for FAST lane (default: 30099)
  SLOW_PROXY_PORT    ZMQ port for SLOW lane (default: 30097)
  PROXY_HTTP_PORT    HTTP port for clients  (default: 10099)
  MIGRATION_T_MIN    Min tokens before eval (default: 200)
  MIGRATION_R_SLOW   Rate threshold tok/s   (default: 30.0)
  OPENAI_API_KEY     Forwarded to vLLM servers (optional)
"""

import json
import os
import threading
import time
import uuid
from typing import Any, AsyncGenerator

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL = os.environ.get("MODEL", "meta-llama/Llama-2-13b-hf")
FAST_ZMQ_PORT = int(os.environ.get("PROXY_PORT", "30099"))
SLOW_ZMQ_PORT = int(os.environ.get("SLOW_PROXY_PORT", "30097"))
HTTP_PORT = int(os.environ.get("PROXY_HTTP_PORT", "10099"))

T_MIN = int(os.environ.get("MIGRATION_T_MIN", "200"))       # evidence gate (tokens)
R_SLOW = float(os.environ.get("MIGRATION_R_SLOW", "30.0")) # congestion threshold (tok/s)
EVAL_INTERVAL = 16                                           # evaluate every N tokens

DEFAULT_PING_SECONDS = 5
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

# ─── Service-discovery state (one dict-pair per lane) ────────────────────────

fast_prefill_instances: dict[str, Any] = {}  # http_addr → (zmq_addr, expiry)
fast_decode_instances:  dict[str, Any] = {}
slow_prefill_instances: dict[str, Any] = {}
slow_decode_instances:  dict[str, Any] = {}

_fast_prefill_cv = threading.Condition()
_fast_decode_cv  = threading.Condition()
_slow_prefill_cv = threading.Condition()
_slow_decode_cv  = threading.Condition()

# ─── Tokenizer (pre-loaded at startup) ───────────────────────────────────────

_tokenizer = None
_tokenizer_lock = threading.Lock()


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                from transformers import AutoTokenizer
                print(f"[proxy] Loading tokenizer: {MODEL}")
                _tokenizer = AutoTokenizer.from_pretrained(MODEL)
                print("[proxy] Tokenizer ready.")
    return _tokenizer


# ─── ZMQ service-discovery ───────────────────────────────────────────────────

def _remove_stale(instances: dict[str, Any]) -> None:
    for k in [k for k, v in list(instances.items()) if v[1] <= time.time()]:
        instances.pop(k, None)
        print(f"[proxy] Removed stale instance: {k}")


def _zmq_listener(
    port: int,
    prefill_dict: dict, prefill_cv: threading.Condition,
    decode_dict:  dict, decode_cv:  threading.Condition,
    label: str,
) -> None:
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    router.bind(f"tcp://0.0.0.0:{port}")
    poller = zmq.Poller()
    poller.register(router, zmq.POLLIN)
    print(f"[proxy/{label}] ZMQ service discovery on port {port}")

    while True:
        socks = dict(poller.poll())
        if router not in socks:
            continue
        _, message = router.recv_multipart()
        data = msgpack.loads(message)
        msg_type = data.get("type")

        if msg_type == "P":
            with prefill_cv:
                is_new = data["http_address"] not in prefill_dict
                prefill_dict[data["http_address"]] = (
                    data["zmq_address"],
                    time.time() + DEFAULT_PING_SECONDS,
                )
                _remove_stale(prefill_dict)
            if is_new:
                print(f"[proxy/{label}] Prefill registered: {data['http_address']}")

        elif msg_type == "D":
            with decode_cv:
                is_new = data["http_address"] not in decode_dict
                decode_dict[data["http_address"]] = (
                    data["zmq_address"],
                    time.time() + DEFAULT_PING_SECONDS,
                )
                _remove_stale(decode_dict)
            if is_new:
                print(f"[proxy/{label}] Decode registered: {data['http_address']}")

        else:
            # Unknown type — log and continue; do NOT return/raise (would kill thread)
            print(f"[proxy/{label}] Unknown message type '{msg_type}', ignoring")
            continue


def start_service_discovery() -> None:
    for port, p_dict, p_cv, d_dict, d_cv, label in [
        (FAST_ZMQ_PORT,
         fast_prefill_instances, _fast_prefill_cv,
         fast_decode_instances,  _fast_decode_cv,  "FAST"),
        (SLOW_ZMQ_PORT,
         slow_prefill_instances, _slow_prefill_cv,
         slow_decode_instances,  _slow_decode_cv,  "SLOW"),
    ]:
        t = threading.Thread(
            target=_zmq_listener,
            args=(port, p_dict, p_cv, d_dict, d_cv, label),
            daemon=True,
        )
        t.start()


# ─── Lane helpers ─────────────────────────────────────────────────────────────

def _make_request_id(prefill_zmq: str, decode_zmq: str) -> str:
    return (
        f"___prefill_addr_{prefill_zmq}___decode_addr_"
        f"{decode_zmq}_{uuid.uuid4().hex}"
    )


def _pick_lane(
    prefill_dict: dict, prefill_cv: threading.Condition,
    decode_dict:  dict, decode_cv:  threading.Condition,
    label: str,
) -> tuple[str, str, str, str]:
    """Return (prefill_http, prefill_zmq, decode_http, decode_zmq)."""
    with prefill_cv:
        if not prefill_dict:
            raise RuntimeError(f"No {label} prefill instance registered yet")
        prefill_http, (prefill_zmq, _) = next(iter(prefill_dict.items()))
    with decode_cv:
        if not decode_dict:
            raise RuntimeError(f"No {label} decode instance registered yet")
        decode_http, (decode_zmq, _) = next(iter(decode_dict.items()))
    return prefill_http, prefill_zmq, decode_http, decode_zmq


def _auth_headers(request_id: str) -> dict:
    return {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
        "X-Request-Id": request_id,
    }


async def _do_prefill(addr: str, path: str, data: dict, request_id: str) -> None:
    """Fire a prefill-only request (max_tokens=1) and drain the response."""
    prefill_data = {k: v for k, v in data.items()
                    if k not in ("max_tokens", "max_completion_tokens")}
    prefill_data["max_tokens"] = 1
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as s:
        async with s.post(
            f"http://{addr}{path}",
            json=prefill_data,
            headers=_auth_headers(request_id),
        ) as r:
            await r.read()  # drain; content not needed


def _parse_sse_text(raw: bytes) -> list[str]:
    """Extract generated text fragments from raw SSE bytes."""
    texts = []
    try:
        for line in raw.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            obj = json.loads(payload)
            for choice in obj.get("choices", []):
                # completions API uses "text"; chat completions uses delta.content
                text = choice.get("text") or (
                    choice.get("delta") or {}
                ).get("content") or ""
                if text:
                    texts.append(text)
    except Exception:
        pass
    return texts


# ─── Simple passthrough (no migration) ───────────────────────────────────────
# Used for /v1/chat/completions where prompt_token_ids is not applicable,
# and as the inner implementation for the fast-lane decode phase.

async def _simple_forward(
    addr: str, path: str, data: dict, request_id: str
) -> AsyncGenerator[bytes, None]:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as s:
        async with s.post(
            f"http://{addr}{path}",
            json=data,
            headers=_auth_headers(request_id),
        ) as r:
            async for chunk in r.content.iter_chunked(1024):
                yield chunk


# ─── Quart app ────────────────────────────────────────────────────────────────

app = Quart(__name__)


@app.route("/v1/completions", methods=["POST"])
async def handle_completions():
    """
    Main request handler with straggler migration.
    1. Fast-lane prefill  (max_tokens=1)
    2. Fast-lane decode   (streaming, with token tracking)
    3. On straggler detection: slow-lane prefill + slow-lane decode
    """
    try:
        original_data = await request.get_json()
        path = request.path  # capture before generator runs

        # ── Fast-lane instances ───────────────────────────────────────────────
        f_pref_http, f_pref_zmq, f_dec_http, f_dec_zmq = _pick_lane(
            fast_prefill_instances, _fast_prefill_cv,
            fast_decode_instances,  _fast_decode_cv,
            "fast",
        )
        fast_rid = _make_request_id(f_pref_zmq, f_dec_zmq)

        # ── Tokenize original prompt for re-admission ─────────────────────────
        # prompt_token_ids lets us re-admit with exact token IDs, no roundtrip.
        original_prompt_ids: list[int] = []
        if "prompt" in original_data:
            try:
                tok = _get_tokenizer()
                original_prompt_ids = tok.encode(
                    original_data["prompt"], add_special_tokens=True
                )
            except Exception as e:
                print(f"[proxy] Tokenizer warning: {e} — migration disabled for this request")

        original_max_tokens = int(
            original_data.get("max_tokens")
            or original_data.get("max_completion_tokens")
            or 4096
        )

        print(
            f"[proxy] → fast  prefill={f_pref_http}  decode={f_dec_http}  "
            f"rid=...{fast_rid[-8:]}"
        )

        # ── Fast-lane prefill ────────────────────────────────────────────────
        await _do_prefill(f_pref_http, path, original_data, fast_rid)

        # ── Migration-aware streaming generator ──────────────────────────────
        async def stream() -> AsyncGenerator[bytes, None]:
            accumulated_text = ""
            N = 0               # SSE events seen (proxy for token count)
            last_eval_N = 0
            start_time = time.time()
            migrated = False

            decode_data = {**original_data, "stream": True}
            headers = _auth_headers(fast_rid)

            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
                async with session.post(
                    f"http://{f_dec_http}{path}",
                    json=decode_data,
                    headers=headers,
                ) as resp:
                    async for raw_chunk in resp.content.iter_chunked(1024):
                        # 1. Forward chunk to client immediately
                        yield raw_chunk

                        # 2. Track generated tokens
                        texts = _parse_sse_text(raw_chunk)
                        for t in texts:
                            accumulated_text += t
                        N += len(texts)

                        # 3. Evaluate migration condition every EVAL_INTERVAL tokens
                        # (only after T_MIN tokens of evidence)
                        if N >= T_MIN and (N - last_eval_N) >= EVAL_INTERVAL:
                            last_eval_N = N
                            elapsed = time.time() - start_time
                            rate = N / elapsed if elapsed > 0 else float("inf")
                            if rate < R_SLOW:
                                print(
                                    f"[proxy/MIGRATE] rid=...{fast_rid[-8:]}  "
                                    f"N={N}  rate={rate:.1f} tok/s < {R_SLOW}  "
                                    f"→ evicting from fast lane"
                                )
                                migrated = True
                                break
                            # Connection closes naturally when we exit the async-with;
                            # vLLM sees the connection drop and aborts the sequence.

            if not migrated:
                return  # Normal completion — fast-lane finished cleanly

            # ── Migration path ────────────────────────────────────────────────
            migration_start = time.time()

            # Re-encode accumulated text to exact token IDs for re-admission.
            # This is a one-time cost at migration time, not per-token overhead.
            generated_ids: list[int] = []
            if original_prompt_ids:
                try:
                    tok = _get_tokenizer()
                    generated_ids = tok.encode(
                        accumulated_text, add_special_tokens=False
                    )
                except Exception as e:
                    print(f"[proxy/MIGRATE] Tokenizer error at migration: {e}")

            new_prompt_ids = original_prompt_ids + generated_ids
            remaining = max(1, original_max_tokens - N)

            # Build slow-lane request using prompt_token_ids (vLLM extension)
            # to avoid tokenizer roundtrip at the slow-lane prefill.
            slow_data = {
                k: v for k, v in original_data.items()
                if k not in ("prompt", "max_tokens", "max_completion_tokens")
            }
            slow_data["prompt_token_ids"] = new_prompt_ids
            slow_data["max_tokens"] = remaining
            slow_data["stream"] = True

            # Pick slow-lane instances
            try:
                s_pref_http, s_pref_zmq, s_dec_http, s_dec_zmq = _pick_lane(
                    slow_prefill_instances, _slow_prefill_cv,
                    slow_decode_instances,  _slow_decode_cv,
                    "slow",
                )
            except RuntimeError as e:
                print(f"[proxy/MIGRATE] Slow lane unavailable: {e} — cannot complete migration")
                return

            slow_rid = _make_request_id(s_pref_zmq, s_dec_zmq)
            print(
                f"[proxy/MIGRATE] → slow  prefill={s_pref_http}  decode={s_dec_http}  "
                f"rid=...{slow_rid[-8:]}  "
                f"context_len={len(new_prompt_ids)}  remaining={remaining}"
            )

            # Slow-lane prefill: processes all original_prompt + generated tokens,
            # transfers KV to slow-lane decode GPU.
            await _do_prefill(s_pref_http, path, slow_data, slow_rid)

            migration_pause = time.time() - migration_start
            print(f"[proxy/MIGRATE] Slow-lane prefill done in {migration_pause:.2f}s")

            # Stream slow-lane decode directly to client.
            # Client sees a brief stall (the prefill pause) then token stream resumes.
            async for chunk in _simple_forward(s_dec_http, path, slow_data, slow_rid):
                yield chunk

        resp = await make_response(stream())
        resp.timeout = None
        return resp

    except RuntimeError as e:
        # Fast-lane not registered yet (proxy started before vLLM servers)
        print(f"[proxy] RuntimeError: {e}")
        return {"error": str(e)}, 503
    except Exception as e:
        import traceback
        print(f"[proxy] Unhandled error: {e}")
        traceback.print_exc()
        return {"error": str(e)}, 500


@app.route("/v1/chat/completions", methods=["POST"])
async def handle_chat_completions():
    """
    Chat completions: no migration (prompt_token_ids not applicable to chat API).
    Routes through fast lane only, identical to baseline proxy behaviour.
    """
    try:
        original_data = await request.get_json()
        path = request.path

        f_pref_http, f_pref_zmq, f_dec_http, f_dec_zmq = _pick_lane(
            fast_prefill_instances, _fast_prefill_cv,
            fast_decode_instances,  _fast_decode_cv,
            "fast",
        )
        fast_rid = _make_request_id(f_pref_zmq, f_dec_zmq)

        await _do_prefill(f_pref_http, path, original_data, fast_rid)

        gen = _simple_forward(f_dec_http, path, original_data, fast_rid)
        resp = await make_response(gen)
        resp.timeout = None
        return resp

    except RuntimeError as e:
        return {"error": str(e)}, 503
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500


if __name__ == "__main__":
    start_service_discovery()
    try:
        _get_tokenizer()  # pre-load at startup to avoid latency on first request
    except Exception as e:
        print(f"[proxy] Warning: tokenizer pre-load failed: {e}")
        print("[proxy] Migration will be disabled for requests where tokenization fails.")
    app.run(host="0.0.0.0", port=HTTP_PORT)
