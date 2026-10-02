import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
log = Path(os.environ.get("FAKE_OLLAMA_LOG", "ollama-calls.jsonl"))
entry = {"args": args}
if args and args[0] == "create":
    mf = Path(args[args.index("-f") + 1])
    entry.update(cmd="create", name=args[1], modelfile=mf.read_text(encoding="utf-8"))
elif args and args[0] == "rm":
    entry.update(cmd="rm", name=args[1])
with open(log, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(entry) + "\n")
if os.environ.get("FAKE_OLLAMA_FAIL"):
    print("Error: could not connect to ollama server", file=sys.stderr)
    sys.exit(1)
print("success")
