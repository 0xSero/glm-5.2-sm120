#!/usr/bin/env python3
"""Long-context MULTI-TURN intelligence test for GLM-5.2-NVFP4-REAP.
A ~55k-token document with facts buried at varying depths, then a 6-turn dialogue
requiring retrieval, multi-fact arithmetic, temporal reasoning, turn-to-turn
dependency, synthesis, and hallucination resistance.
Streaming, NO max_tokens. Measures TTFT, prefill tok/s, decode tok/s per turn.
"""
import json, time, sys, urllib.request

URL = "http://127.0.0.1:5329/v1/chat/completions"
MODEL = "GLM-5.2-NVFP4-REAP-469B"

FACTS = {
    450:  "EXPEDITION RECORD: Project Meridian departed from Base Camp Theta on day 3 with a crew of exactly 7 members.",
    1050: "PERSONNEL NOTE: Dr. Elena Vasquez, the lead geologist, discovered a new mineral named veridium in rock sample R-42.",
    1500: "SAMPLE LOG: Each veridium crystal weighs 0.8 kg. The team collected 15 veridium crystals from the northern ridge.",
    1950: "LOGISTICS: The supply drop on day 9 delivered 240 ration units. Each crew member consumes 3 ration units per day.",
    2400: "ROSTER CHANGE: On day 12, exactly 2 crew members returned to Base Camp Theta; the remaining crew continued the survey.",
    2700: "CACHE RECORD: The final equipment cache was buried at Site Omega-7, located 1.2 km east of the northern ridge.",
}

def build_document(n_lines=3000):
    out = []
    for i in range(n_lines):
        if i in FACTS:
            out.append(f"[{i:04d}] {FACTS[i]}")
        else:
            t = (i * 7) % 60 - 20
            w = (i * 13) % 55
            g = (i * 29) % 500
            out.append(f"[{i:04d}] Daily log entry {i}: ambient temperature {t}C, wind {w} km/h; routine telemetry archived for survey grid {g}.")
    return "\n".join(out)

DOC = build_document()
DOC_PREAMBLE = (
    "Below is the full operational log for 'Project Meridian'. Read it carefully; "
    "I will ask several questions about it.\n\n=== BEGIN LOG ===\n" + DOC + "\n=== END LOG ===\n"
)

# (question, checker(content_lower) -> bool, expected_note)
TURNS = [
    ("Who is the lead geologist, and what did she discover?",
     lambda c: "vasquez" in c and "veridium" in c,
     "Dr. Elena Vasquez discovered veridium"),
    ("What is the total mass, in kg, of all the veridium collected from the northern ridge? Show the calculation.",
     lambda c: "12" in c and ("0.8" in c or "15" in c),
     "15 x 0.8 = 12 kg"),
    ("Using the day-9 supply drop and the crew size at that point in the expedition, how many full days of rations did that drop provide? Account for the crew size as of day 9. Show your work.",
     lambda c: ("11.4" in c or "11.43" in c or "11 " in c or "21" in c),
     "240 / (7*3) = 240/21 = ~11.43 days (still 7 crew on day 9)"),
    ("After the day-12 roster change, what is the crew's new total daily ration consumption rate in units per day?",
     lambda c: "15" in c and ("5" in c),
     "7-2 = 5 crew; 5*3 = 15 units/day"),
    ("In exactly 3 sentences, summarize the expedition, and state the location of the final cache relative to the northern ridge.",
     lambda c: "1.2" in c and "east" in c,
     "summary + cache 1.2 km east of northern ridge"),
    ("Does the log state the date on which the expedition returned to Base Camp Theta? Answer truthfully based only on the log.",
     lambda c: ("no" in c or "not" in c or "doesn't" in c or "does not" in c) and ("return" in c or "state" in c or "mention" in c or "specif" in c),
     "NOT stated (hallucination check)"),
]

def stream_turn(messages):
    payload = {"model": MODEL, "messages": messages, "temperature": 0.2,
               "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_last = t0
    reasoning, content = [], []
    ptoks = ctoks = None; finish = None
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
                ptoks = obj["usage"].get("prompt_tokens", ptoks)
                ctoks = obj["usage"].get("completion_tokens", ctoks)
            for ch in obj.get("choices", []):
                d = ch.get("delta", {}) or {}
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                if piece:
                    if t_first is None:
                        t_first = time.time()
                    t_last = time.time()
                    (content if d.get("content") else reasoning).append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    ttft = (t_first - t0) if t_first else None
    span = (t_last - t_first) if (t_first and t_last > t_first) else None
    return {
        "ttft": ttft,
        "prefill_tps": (ptoks / ttft) if (ttft and ptoks) else None,
        "decode_tps": ((ctoks - 1) / span) if (span and ctoks) else None,
        "ptoks": ptoks, "ctoks": ctoks, "finish": finish,
        "reasoning": "".join(reasoning), "content": "".join(content),
    }

def main():
    messages = []
    results = []
    for idx, (q, check, expected) in enumerate(TURNS, 1):
        user_text = (DOC_PREAMBLE + "\n" + q) if idx == 1 else q
        messages.append({"role": "user", "content": user_text})
        r = stream_turn(messages)
        ans = r["content"] or r["reasoning"]
        messages.append({"role": "assistant", "content": ans})
        passed = check((ans or "").lower())
        r["passed"] = passed
        results.append(r)
        print("=" * 80)
        print(f"TURN {idx}  prompt={r['ptoks']} completion={r['ctoks']} finish={r['finish']}  -> {'PASS' if passed else 'FAIL'}")
        ttft = f"{r['ttft']*1000:.0f}ms" if r['ttft'] else "n/a"
        pf = f"{r['prefill_tps']:.0f}" if r['prefill_tps'] else "n/a"
        dc = f"{r['decode_tps']:.1f}" if r['decode_tps'] else "n/a"
        print(f"  TTFT={ttft}  prefill={pf} tok/s  decode={dc} tok/s")
        print(f"  Q: {q}")
        print(f"  expected: {expected}")
        print(f"  ANSWER: {(r['content'] or '[in reasoning] '+r['reasoning'])[:900]}")
        sys.stdout.flush()
    print("\n" + "#" * 80 + "\nLONG-CONTEXT MULTI-TURN SUMMARY\n" + "#" * 80)
    print(f"{'turn':5} {'prompt':>7} {'compl':>6} {'TTFT':>9} {'prefill':>9} {'decode':>8} {'finish':>7} {'verdict':>8}")
    npass = 0
    for i, r in enumerate(results, 1):
        npass += 1 if r["passed"] else 0
        ttft = f"{r['ttft']*1000:.0f}ms" if r['ttft'] else "n/a"
        pf = f"{r['prefill_tps']:.0f}" if r['prefill_tps'] else "n/a"
        dc = f"{r['decode_tps']:.1f}" if r['decode_tps'] else "n/a"
        print(f"{i:5} {str(r['ptoks']):>7} {str(r['ctoks']):>6} {ttft:>9} {pf:>9} {dc:>8} {str(r['finish']):>7} {'PASS' if r['passed'] else 'FAIL':>8}")
    print(f"\nVERDICT: {npass}/{len(results)} turns passed.  Final-turn context: {results[-1]['ptoks']} tokens.")

if __name__ == "__main__":
    main()
