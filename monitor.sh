#!/usr/bin/env bash
# Forward GLM-5.2 container logs to a file AND persist peak prefill/decode tok/s.
# Survives container auto-restarts (re-attaches). Started in the background by
# launch.sh; also runnable standalone:  ./monitor.sh &
#
#   logs/serve.log  — forwarded vLLM serve logs (rotated at 100 MB -> serve.log.1)
#   logs/peak.json  — { peak_prefill_tps, peak_decode_tps, updated }
set -uo pipefail
cd "$(dirname "$0")"
[ -f .env ] && { set -a; . ./.env; set +a; }
export NAME="${NAME:-glm52-vllm}"
export LOG_DIR="${LOG_DIR:-$(pwd)/logs}"
mkdir -p "$LOG_DIR"

exec python3 -u - <<'PY'
import os, sys, json, re, subprocess, time
name=os.environ["NAME"]; d=os.environ["LOG_DIR"]
serve=os.path.join(d,"serve.log"); peakp=os.path.join(d,"peak.json")
MAXBYTES=100*1024*1024
# vLLM prints this every ~10s; the window-average peak is our "peak decode/prefill".
rx=re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+)")
try: cur=json.load(open(peakp))
except Exception: cur={}
pp=float(cur.get("peak_prefill_tps",0.0)); dp=float(cur.get("peak_decode_tps",0.0))
def savepeak():
    json.dump({"peak_prefill_tps":round(pp,1),"peak_decode_tps":round(dp,1),
               "updated":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},
              open(peakp,"w")); 
savepeak()
f=open(serve,"a")
def write(line):
    global f
    f.write(line); f.flush()
    if f.tell()>MAXBYTES:
        f.close(); os.replace(serve, serve+".1"); f=open(serve,"a")
while True:
    try:
        p=subprocess.Popen(["docker","logs","-f","--since","2s",name],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except Exception:
        time.sleep(3); continue
    for line in p.stdout:
        write(line)
        m=rx.search(line)
        if m:
            a=float(m.group(1)); b=float(m.group(2)); ch=False
            if a>pp: pp=a; ch=True
            if b>dp: dp=b; ch=True
            if ch: savepeak()
    p.wait()
    time.sleep(2)   # container restarting -> reattach
PY
