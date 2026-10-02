# Working on Pygmalion's Hoard

A local model tuning and extension studio (package `pygmalion_hoard`, service `pygmalion-hoard`, app id `pygmalion`, port 5202). It builds datasets, trains LoRA and QLoRA adapters, merges, extends context, converts, quantizes, evaluates through Galton's Hoard and publishes. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first.

## Commands

```bash
python -m pytest -q                       # the whole suite; no network, no GPU, no real torch
python scripts/gen_api_doc.py             # regenerate docs/API.md after changing any tool (a test fails when it is stale)
npx vite build                            # rebuild the UI into pygmalion_hoard/static (run from the repository root)
python -m pygmalion_hoard                 # run on http://127.0.0.1:5202 (PYGMALION_DATA_DIR=<folder> for a scratch copy)
```

To try the whole app without a GPU, run it with `PYGMALION_WORKERS_DIR=tests/fake_workers PYGMALION_OFFLINE=1`: training and downloads then use the fake workers.

## Rules of the code

- Every state change goes through `Services`/`Ops`; the store only stores. The UI, the REST agent route and MCP all call the same handlers in `agent_tools.py`: add a capability there once and the three surfaces get it. The bundled UI calls tools through `POST /api/ui/call`.
- A new tool needs: a pydantic argument model with descriptions, a first description line of at most 110 characters (English, then a Spanish phrase), synonyms, annotations, a test in `tests/test_tools.py`, a regenerated `docs/API.md` and an entry in the README tool lists. Deletes need `confirm=true`.
- GPU work only through a lease from the family hub on an allowed GPU (`gpus.allowed`, default 2 and 3). Never touch another GPU. Allowing a reserved GPU needs `confirm_reserved`, which only the owner may decide.
- Workers (`pygmalion_hoard/workers/*.py`) run in the trainer environment, never import the app, speak JSON lines on stdout and stop on SIGTERM, SIGINT or SIGBREAK after saving a checkpoint. Keep them plain and robust; the app cannot import torch.
- Adapters, models, datasets and anything a model wrote are untrusted: read only safetensors and JSON, never unpickle, never execute.
- Galton is reached through `evaluate.py` only, with the tool names `galton_status`, `suites_list`, `suite_create`, `suite_remove`, `cases_import`, `run_start`, `run_status`, `run_cancel`, `compare` and `run_results`. Evaluation holds no GPU lease. The held-out suite of a dataset version is named `pyg-<dataset>-v<n>-eval` and carries `sha256:<hash>:<cases>` in its description, which is how it is recognised and reused; a suite with that name that this app did not build is never overwritten or removed.
- What an evaluation compares with is decided in `reference.py` only: for the intents that ask whether a training helped, the base model the training started from in the result's own quantization and calibration (never the f16 file of the same fine-tune), prepared by a pipeline when it does not exist; the parent file only for a plain quantization or `intent=quant`; `against` always wins. Never pick a reference with `nearest_comparable` alone.
- Every thread the app starts is stopped and joined by `Services.stop` (job lanes, lease helpers, the event worker) and is named `pygmalion-*`; never start a bare daemon thread that outlives a test, and never call `family.emit` from app code (it starts a thread per event): use `Services.emit`. The suite fails a test that leaves a `pygmalion-*` or `hoard-*` thread running. A new thread needs a stop path, a timeout on its join and a line in `tests/test_shutdown.py`.
- Secrets (the Hugging Face token) are write-only and redacted in logs and job output.
- `hoard_link/` is vendored and byte-identical to upstream: never edit it here. The start-up, guard, error envelope, PWA files, health route, SQLite layer, agent kit and bridge, atomic writes, ids, child processes, path rules, PII scanner and chunker come from it; do not write a private copy of any of them. A file the app or a worker writes goes through `hoard_link.atomic` (workers through `workers/_atomic.py`).
- User-facing text of the interface goes in `client/src/i18n.js` (Spanish first, English second). Anything the backend can show (errors, hints, warnings, notes, plan lines, progress messages, job titles) is a key in `pygmalion_hoard/messages.py` plus an entry in `client/src/msgs.js`; never raise `PygmalionError` with a free sentence and never print a backend field in the client without `t.msg`/`t.error`/`t.hint`. Words that clash with `PygmalionError`'s own arguments (`code`, `status`) are not used as parameter names in the catalogue. Spanish is castellano de España. Keep the UI and the docs plain: no marketing lines. `tests/test_client.py` checks that every key used exists in both languages.
- Tests use invented data and fake backends (`tests/fake_workers`, `tests/fake_tools`, `tests/helpers.py`); never add real names, accounts or paths. Time is injected where it matters.
- Tests must not depend on the computer: `hermetic_host` (autouse, `tests/conftest.py`) clears the PYGMALION_/OLLAMA_/HF_/HOARD_ variables, points the Windows default folders, `procs.which` and the Ollama install folders at nowhere, and `tests/test_hermetic.py` checks that the real defaults still apply when something is there. A test that needs one of them sets it itself. Path rules are tested on `PureWindowsPath` so Linux catches Windows separators; fake programs are POSIX scripts, so tests that run them carry `@needs_posix` (`PYG_TEST_PLACEHOLDER_TOOLS=1` runs the suite on Linux with the inert programs Windows gets).
- Everything must run on Windows (paths with spaces and apostrophes, `CREATE_NO_WINDOW`) and on Linux.

## Where things are

| Need | Look at |
|---|---|
| A job does not start or lingers | `jobs.py` (lanes, states, pipelines, `$ref` resolution), `runners.py` |
| The run ends with a fatal error at exit, or a thread outlives a test | `Services.stop`, `JobManager.stop`, `events.py`, `GpuManager.close`, `tests/test_shutdown.py`, the `no_lingering_threads` fixture |
| A memory or time estimate looks wrong | `vram.py`, `planner.py` (time model: constants at the top of `vram.py`) |
| A sentence is in the wrong language | `messages.py`, `client/src/msgs.js`, `tests/test_messages.py` |
| A dataset source or operation | `datasets/sources.py`, `datasets/operations.py`, `datasets/service.py` |
| A chat is masked wrongly for training | `chatmask.py`, `workers/train_lora.py` |
| A merge result is wrong | `merge_math.py`, `workers/merge_models.py`, `workers/merge_lora.py` |
| GGUF conversion or quantization | `llama_tools.py`, `convert_detect.py`, `gguf_meta.py` |
| Galton verdicts | `evaluate.py` |
| What a result is compared with, the base file to prepare | `reference.py`, `Operations.evaluate_start` |
| Publishing | `publish.py` |
| The UI | `client/src/pages/*.jsx`, `client/src/components/*`, `client/src/i18n.js` |

## Before finishing a change

1. `python -m pytest -q` is green.
2. `python scripts/gen_api_doc.py --check` passes.
3. If the UI changed: `npx vite build`, then open the running app and look at every page in Spanish and English, at desktop and phone width.
4. README.md, README.es.md and docs/ARCHITECTURE.md say what the app can and cannot do now.
