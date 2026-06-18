#!/usr/bin/env python3
"""GLM-5.2-NVFP4-REAP coherence + perf battery.
Streaming, NO max_tokens (model finishes naturally). Measures TTFT, prefill tok/s, decode tok/s.
"""
import json, time, sys, urllib.request

URL = "http://127.0.0.1:5329/v1/chat/completions"
MODEL = "GLM-5.2-NVFP4-REAP-469B"

def run_chat(messages, temperature=0.6, label=""):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = None
    t_last = t0
    reasoning, content = [], []
    prompt_toks = completion_toks = None
    finish = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("usage"):
                u = obj["usage"]
                prompt_toks = u.get("prompt_tokens", prompt_toks)
                completion_toks = u.get("completion_tokens", completion_toks)
            for ch in obj.get("choices", []):
                d = ch.get("delta", {}) or {}
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                if piece:
                    if t_first is None:
                        t_first = time.time()
                    t_last = time.time()
                    if d.get("content"):
                        content.append(d["content"])
                    else:
                        reasoning.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    wall = time.time() - t0
    ttft = (t_first - t0) if t_first else None
    decode_span = (t_last - t_first) if (t_first and t_last > t_first) else None
    decode_tps = (completion_toks - 1) / decode_span if (decode_span and completion_toks) else None
    prefill_tps = (prompt_toks / ttft) if (ttft and prompt_toks) else None
    return {
        "label": label, "ttft": ttft, "prefill_tps": prefill_tps, "decode_tps": decode_tps,
        "prompt_toks": prompt_toks, "completion_toks": completion_toks, "finish": finish,
        "wall": wall, "reasoning": "".join(reasoning), "content": "".join(content),
    }

def show(r):
    print("=" * 78)
    print(f"[{r['label']}]  finish={r['finish']}  prompt={r['prompt_toks']} completion={r['completion_toks']} wall={r['wall']:.1f}s")
    ttft = f"{r['ttft']*1000:.0f}ms" if r['ttft'] else "n/a"
    pf = f"{r['prefill_tps']:.0f}" if r['prefill_tps'] else "n/a"
    dc = f"{r['decode_tps']:.1f}" if r['decode_tps'] else "n/a"
    print(f"    TTFT={ttft}   prefill={pf} tok/s   decode={dc} tok/s")
    if r["reasoning"]:
        rr = r["reasoning"]
        print(f"    --- reasoning ({len(rr)} chars) ---\n{rr[:700]}{' ...[snip]' if len(rr)>700 else ''}")
    if r["content"]:
        cc = r["content"]
        print(f"    --- answer ({len(cc)} chars) ---\n{cc[:1500]}{' ...[snip]' if len(cc)>1500 else ''}")
    sys.stdout.flush()

def needle_prompt(target_lines=3200, needle_at=1850):
    lines = []
    needle = "IMPORTANT FACT: The secret passphrase for the vault is BLUE-WALRUS-7421-OMEGA."
    for i in range(target_lines):
        if i == needle_at:
            lines.append(f"Line {i}: {needle}")
        else:
            lines.append(f"Line {i}: The survey team recorded ambient readings and logged routine telemetry for sector {i%97}.")
    body = "\n".join(lines)
    q = "\n\nQuestion: Somewhere above is a single line beginning with 'IMPORTANT FACT'. What is the secret passphrase for the vault? Reply with only the passphrase."
    return body + q

TESTS = {
    "logic": [{"role": "user", "content": "All Bloops are Razzies. All Razzies are Lazzies. Are all Bloops Lazzies? Answer YES or NO, then explain the transitive reasoning."}],
    "reasoning_math": [{"role": "user", "content": "A train leaves city A at 9:00 traveling 60 mph toward city B, 300 miles away. Another leaves B at 10:00 traveling 90 mph toward A. At what clock time do they meet? Show your work."}],
    "code": [{"role": "user", "content": "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals, with a docstring and a couple of example assertions. Then briefly explain the time complexity."}],
    "philosophy": [{"role": "user", "content": "Is mathematics discovered or invented? Give a reasoned position, acknowledge the strongest counterargument, and conclude."}],
    "ascii": [{"role": "user", "content": "Draw an ASCII-art cat sitting next to a fish bowl. Use a code block."}],
}

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "core"
    results = []
    if which in ("core", "all"):
        for name, msgs in TESTS.items():
            r = run_chat(msgs, label=name); show(r); results.append(r)
        # multi-turn
        conv = [{"role": "user", "content": "I'm thinking of a number: 42. Remember it. Also my favorite color is teal."}]
        r1 = run_chat(conv, label="multiturn_t1"); show(r1); results.append(r1)
        conv.append({"role": "assistant", "content": r1["content"] or r1["reasoning"][:200]})
        conv.append({"role": "user", "content": "Multiply the number I gave you by 3, then tell me my favorite color."})
        r2 = run_chat(conv, label="multiturn_t2"); show(r2); results.append(r2)
    if which in ("long", "all"):
        p = needle_prompt()
        r = run_chat([{"role": "user", "content": p}], temperature=0.0, label="long_context_needle"); show(r); results.append(r)
    print("\n" + "#" * 78 + "\nSUMMARY\n" + "#" * 78)
    print(f"{'test':22} {'TTFT':>8} {'prefill':>10} {'decode':>9} {'prompt':>7} {'compl':>7} {'finish':>8}")
    for r in results:
        ttft = f"{r['ttft']*1000:.0f}ms" if r['ttft'] else "n/a"
        pf = f"{r['prefill_tps']:.0f}" if r['prefill_tps'] else "n/a"
        dc = f"{r['decode_tps']:.1f}" if r['decode_tps'] else "n/a"
        print(f"{r['label']:22} {ttft:>8} {pf:>10} {dc:>9} {str(r['prompt_toks']):>7} {str(r['completion_toks']):>7} {str(r['finish']):>8}")

if __name__ == "__main__":
    main()
