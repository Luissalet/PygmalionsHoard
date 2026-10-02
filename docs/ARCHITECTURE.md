# Architecture

```
 UI (React)  ──POST /api/ui/call──┐
 assistants ─ /api/agent/call ────┼──► agent_tools.py: one catalogue (44 tools) ──► Services / Ops
 mcp_server.py (stdio bridge) ────┘                                                   │
                                                                                      ▼
 datasets/ (sources, operations, service) ◄── planner.py (parameters, memory, pipelines) ──► jobs.py (store-backed queue)
                                                                                      │  lanes: gpu · cpu, one job each
                                                                                      ▼
                                   runners.py ── workers/*.py in the trainer environment (JSON lines) ── llama.cpp tools (llama_tools.py)
                                      │                 │
                                      │                 └── GPU lease from the family hub (gpus.py, hoard_link.lease)
                                      ▼
                          lineage.py (artifacts, recipes) ◄── evaluate.py (Galton) · publish.py (Ollama, llama.cpp backends)
```

## Layout

| Path | Role |
|---|---|
| `pygmalion_hoard/main.py`, `__main__.py` | App factory (shared guard, error envelope, health, PWA files and single-page server from `hoard_link.service`) and the shared launcher `run_main` (one running copy, rotating `data/logs/pygmalion-hoard.log`, works without a console) |
| `config.py`, `settings.py` | Process settings from the environment (`PYGMALION_*`, `appconfig`) and `.env` (read by the shared parser, never exported); settings stored in the database as texts (the shared database stores JSON; an older plain-text row reads the same) |
| `db.py`, `store.py` | Schema and migrations over the shared `sqlkit.Database` (SQLite, WAL, busy timeout, re-entrant transactions); every query. New ids are lowercase ULIDs (`util.new_id`); older ids still resolve |
| `services.py` | Wires everything from a `Config`; dashboard and views |
| `ops.py` | The operations the tools call: downloads, dataset builds, pipelines |
| `agent_tools.py` | The tool catalogue: name, description, pydantic arguments, annotations, handler; `Tool`, the catalogue, `call_tool` and the result cap are the shared `agentkit` |
| `api/` | REST: `agent.py` (the shared agent router: assistants, Bearer token), `ui.py` (dashboard, UI calls), `health.py` (`/api/status`; `/api/health` is the shared route) |
| `jobs.py` | Job manager: two lanes, states, pipelines with `$ref` resolution, cancel, resume, interrupted-on-restart |
| `runners.py` | One runner per job kind; builds the worker arguments and registers the artifacts |
| `workers/` | Run in the trainer environment: `train_lora`, `merge_lora`, `merge_models`, `ctx_extend`, `download`, `probe_env`; shared `_common`, `_stio` (safetensors streaming) and `_atomic` (loads the vendored `hoard_link/atomic.py` by its path, so state files, checkpoints and safetensors are replaced with the Windows-safe retry) |
| `procs.py` | Subprocesses on the shared `hoard_link.proc` (start, Ctrl+Break request, tree kill): the stop file, log file, environment, redaction; the output reader never outlives the program (a child that keeps the pipe open gets its group killed) |
| `events.py` | `EventPump`: events for the family bus go through a queue to one worker thread that `Services.stop` joins (instead of one daemon thread per event) |
| `gpus.py`, `vram.py` | Allowed GPUs, leases, inventory; memory estimate and the time model (throughput constants, start-up, per-step overhead) |
| `messages.py`, `errors.py` | The message catalogue (`ERRORS`, `TEXTS`), `CodedText`, recognition of stored sentences, `wire` and `localise` for the UI; `PygmalionError` |
| `planner.py` | Training parameters, the plan, the steps after a stage, the baseline for evaluation |
| `datasets/` | Source loaders, operations (dedupe, MinHash, length, language, PII), versions, statistics; PII is the shared `idcheck.scan_pii`, chunking the shared `docs.chunking`, text decoding the shared `docs.textclean.decode_text` |
| `chatmask.py` | Which tokens of a chat the loss covers (pure, testable without torch) |
| `merge_math.py` | Linear, SLERP, TIES and DARE on numpy arrays |
| `gguf_meta.py`, `convert_detect.py`, `llama_tools.py` | GGUF metadata reader, architecture support check (scans `convert_hf_to_gguf.py` and the `conversion/` package for `@ModelBase.register(...)`, one or several names, one or several lines), llama.cpp command lines and output parsing |
| `lineage.py` | Artifacts, ancestors, graph layout, recipe export, safe deletion |
| `reference.py` | What a result is measured against: the base model's file in the result's quantization (steps to build it, what is reused), the model before a context extension, or the parent file |
| `evaluate.py` | Galton client (hub first, direct fallback), the evaluation run, verdict summary, needle table |
| `publish.py` | Modelfile and `ollama create`, llama.cpp backend entries, unpublishing |
| `hf.py`, `envcheck.py`, `workdir.py`, `paths.py` | Hugging Face search and file lists, environment check, work folder layout, path safety (`paths.py` words the shared `hoard_link.paths` reasons with the message catalogue) |
| `errors.py` | The single error class (code, message, hint), an `AppError` of the shared kit |
| `hoard_link/` | Vendored family library (event bus, calls to other apps, GPU leases). Not edited here |
| `mcp_server.py` | stdio bridge (the shared `CatalogBridge`): proxies to the running app, starts it when needed |
| `client/` | React 19 + Vite 6 + Tailwind 4 UI; built into `pygmalion_hoard/static` |

## Jobs and pipelines

A job has a kind, a lane (`gpu` for training, the importance matrix, perplexity and evaluation, the last of which holds no lease because Galton does the work; `cpu` for the rest), parameters and a state: `queued`, `waiting_gpu`, `running`, `done`, `failed`, `cancelled`, `interrupted`. Each lane runs one job at a time, so a CPU merge can run while a training run holds a GPU. `status()` takes what a lane runs from the database (a row that is no longer active is idle) and first closes, as `interrupted`, the active rows that nothing in this process runs (`reap_stale`); the lane loop survives errors in its bookkeeping and always frees the lane. The Trabajos badge counts running and waiting jobs only. `pipeline_steps` lists every step of a chain, the planned ones included (no id yet, state `planned`), and the job page polls while the chain is live.

Program output goes through `logclean.py` before it is stored: escape sequences are removed and a progress or spinner frame replaces the previous frame of the same activity, in the log file as it is written and again when an older log is read.

A pipeline is a list of jobs linked by `then`. Later steps refer to earlier results with `$ref` strings that are resolved when the step starts: `$adapter`, `$merged`, `$ctx_variant`, `$gguf` (the unquantized file), `$imatrix`, `$quant`, `$quant:Q4_K_M`, `$prev`, and aliases set with `_as`. When evaluation is requested, the steps that build the base model's GGUF (`Reference.chain`) are put before the training step, so the comparison has a baseline at the same quantization. If a step fails or is cancelled, the rest is not started; resuming the failed step continues the pipeline.

Events: `jobevents.JobEvents` (owned by the job manager) emits `pygmalion.job.queued` on submit, `started` when the lane claims the job, `progress` (from the job's `pct`, with `eta_s` and the GPU it holds; at most one every 5 seconds), and `done`, `failed` or `cancelled` at the end. A job that was running when the app stopped is reported `failed` once at the next start (it stays `interrupted` and resumable here). Data: `{job_id, title, kind, job_kind, progress, gpu, eta_s, url, error, pipeline}`. `kind` is the canonical kind (`merge_lora` and `merge_models` are both `merge`; `train`, `convert`, `quantize` and `publish` are as they are, the other kinds keep their name) and `job_kind` is this app's own. A finished `publish` job also carries `model`, the Ollama tag it created (the llama.cpp backend id when that was the only target). The old `pygmalion.job_queued` and `pygmalion.job_done` are no longer emitted; the hub maps them onto the canonical names.

GPU jobs estimate memory first, then take a lease from the hub for it on an allowed GPU and run the worker with `CUDA_VISIBLE_DEVICES` set to the granted card (and `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the index means the same card as in nvidia-smi). While waiting the job is `waiting_gpu` with the hub's queue position. The lease is released when the worker ends, whatever the result.

## Threads and shutdown

The app starts a fixed set of threads, and every one of them is stopped and joined by `Services.stop()` (called by the lifespan of the app and by the test fixtures), each with a timeout, in this order: the job lanes (`pygmalion-lane-gpu`, `pygmalion-lane-cpu`: `JobManager.stop` cancels what runs, wakes the lanes and joins them under one shared deadline; a lane that is leaving never claims a job), the lease helpers (`pygmalion-lease`: `GpuManager.close` abandons the requests that wait; a lease granted afterwards is released at once), the event worker (`pygmalion-events`: `EventPump.close` drops what waits and joins), the language-model link and the database. `stop()` returns the names of the threads that did not end in time (empty when clean) and logs them; calling it twice is harmless. An output reader (`pygmalion-proc-reader`) ends with its program: when a child keeps the pipe open the group is killed.

Why it matters: a daemon thread that is still running when the interpreter shuts down can be killed in the middle of a write or of a call into a C extension, and the process then ends with a fatal error (a dump of the loaded extension modules, numpy among them) after all the work was done. `family.emit` of the vendored link starts a new daemon thread per event, so the app does not call it from its own threads; it queues events on `EventPump` and the worker calls `family.emit(block=True)`. The app code itself never imports numpy (the merge arithmetic runs in the worker processes), which `tests/test_shutdown.py` checks in a fresh interpreter.

Tests: `tests/conftest.py` has an autouse fixture that fails the test that leaves a thread named `pygmalion-*` or `hoard-*` running (it waits up to 5 s for it to end first); `tests/test_shutdown.py` starts and stops live services repeatedly and checks that no thread, daemon or not, remains.

## Workers

A worker is a script run by the trainer environment's Python with `--args args.json`. It prints one JSON object per line (`progress`, `eval`, `artifact`, `result`, `log`) and never imports the app. The app stops it by creating the stop file named in `PYGMALION_STOP_FILE` (the job folder's `STOP`) and sending a signal (a worker started without a console on Windows only sees the file); the worker saves a checkpoint and exits, and the job can be resumed from the last checkpoint. Tests replace the workers with `tests/fake_workers/*.py` that print realistic lines.

`merge_lora.py` computes `W + (alpha / r) B A` one tensor at a time in float32 over the base's safetensors shards and copies every other file of the base unchanged; adapters it cannot map (DoRA, extra saved modules, unmatched names) go through transformers and PEFT instead.

## Evaluation

What is compared with what. `reference.py` decides, from the intent and the lineage of the result: an explicit `against` always wins; for `dataset`, `style`, `writing`, `code`, `general` and `smoke` (the questions "did the training help or hurt?") the reference is the model the nearest training started from (the first parent of the nearest `adapter` ancestor), as a GGUF written like the result; for `context` it is the model before the nearest `ctx_variant`; for `quant`, and by default for a result with no training in its ancestry, it is the nearest GGUF ancestor (the f16 file the quantization came from). "Written like the result" (`Reference.spec_of`) is the quantization type (or the unquantized `outtype`) and, when the quantized file has an `imatrix` parent, that matrix's calibration: its tag (`calib.tag_of`, the hash of the calibration request, stored on the matrix artifact), its request and its chunk count. `Reference.chain` looks among the plain GGUF files below the base model (no adapter, no dataset version, file still on disk) for one with the same type and the same matrix tag (or none), reuses an existing unquantized file and an existing matrix of the same tag, and returns the missing `convert`, `imatrix` and `quantize` steps with `_as` names (`baseline_gguf`, `baseline_imatrix`, `baseline`). `Operations.evaluate_start` queues those steps and the evaluation as one pipeline (`against` is `$baseline`); the new files are ordinary artifacts below the base model, so the next evaluation, the perplexity tables and `with_baseline` of the training pipelines find them. `Evaluator.run` without a ready reference raises `eval_reference_missing` instead of falling back to another file, records `reference` (`mode`, `model`, `quant`, `imatrix`) with the verdict, and appends the plan's warnings (for example `eval_ref_unknown_base`, when the result has a training behind it but its base model is not in the lineage and the parent file is used). `evaluate_plan` (also `artifact_get["evaluate_plan"]`) is `Evaluator.explain`: the plan without queueing anything, with the coded sentence the Evaluate box shows. A merge of whole models has no training base and uses the parent file.

`evaluate.py` starts a Galton run with the reference and the child as contestants, polls its status, asks for the comparison of the child against the parent and stores the verdict, the difference with its interval and the counts on the child artifact. For context variants it also reads the long-context suite's results and builds the table of scores per length. The call goes through the family hub first and falls back to Galton's own agent route with its token.

Held-out records. A result descended from a training job whose dataset version has an `eval` split is evaluated with `intent: dataset` by default. `Evaluator.ensure_suite` turns each held-out record into a case (`record_case`: chat records ask everything before the last assistant turn, instruction records ask the prompt, text records ask for the continuation of their first 60 %), with the expected answer twice: as the case's reference and inside a `judge` checker whose rubric asks for the same content as the reference and penalises invented data. It lists Galton's own suites, reuses the one whose description carries `sha256:<version hash>:<cases>` and the right number of cases, replaces one built from this version that was left half imported, and picks another name (`...-<hash>`) rather than touch a suite it did not build. The run covers that suite plus the general `rapida` suite; the verdict on the held-out suite and the one on `rapida` are both stored (`dataset`, `regression`), and `promote_suggested` is false when the general suite got worse. Before building anything it reads Galton's `judge` setting: no judge model stops the evaluation with a hint, and answers a run reports as waiting for the judge are a warning, or an error when they leave no verdict.

Perplexity. `perplexity` jobs store `{value, error, ctx, chunks, text}` on the measured artifact (read back after the write, and merged under the database lock so concurrent measurements do not overwrite each other). The planner measures the unquantized file and every quantization with identical text, context and chunks, plus the base model's own files when they exist or are built for the evaluation. `Lineage.annotate_ppl` adds `ppl_cmp` (against the nearest measured ancestor and against the base's file of the same quantization) and `ppl_family` lists the comparable files of a family; numbers measured differently are never compared.

## Lineage and recipes

Every artifact stores its parents, the job that made it, the dataset version hash and its metrics (training summary, perplexity, GGUF metadata, Galton record). `artifact_recipe` walks the ancestors from the base and lists the steps in order with portable paths. Deleting an artifact never deletes a downloaded base or an Ollama tag's files, and only removes files inside the work folder.

Publishing writes the Modelfile `num_ctx` from `publish.num_ctx` (8192), clamped to the trained context read from the GGUF, so Ollama does not reserve the cache of a 256k window; a context variant is published at its own length and a value asked for in the form is kept.

## Client

Hash-routed single page with ten pages. Every call goes through `api.call(name, args)` to `/api/ui/call`. The dashboard is polled every 3 seconds while jobs are active and every 30 seconds otherwise; `usePoll` makes its first call at once whatever the visibility of the tab, and the app bumps a version (which reloads every page) whenever the dashboard counts change, so a write or an environment check shows at once. The service worker is generated by the shared `install_pwa` with a cache name taken from the build (the hash of `index.html`); it fetches the page from the network first and is served with `Cache-Control: no-cache`, so a new build is never hidden by an older one; the brand row is `<img src="/icon-192.png">`. The base-model count is read from the disk (`services.sync_bases`) like `bases_list`, never from a stale cache. Texts of the interface live in `client/src/i18n.js`; the messages of the backend live in `pygmalion_hoard/messages.py` and `client/src/msgs.js` (next section).

## Messages and languages

Every sentence the backend can show is an entry of `messages.py`: `ERRORS` (`key -> (message, hint)`) and `TEXTS` (`key -> sentence`), with bare `{name}` placeholders. Code raises `PygmalionError(code, "key", **params)` or builds `text("key", **params)`; both keep the English sentence (what an assistant reads) and remember `key` and `params`. Workers (`WorkerError("key", **params)`) load the same file by path and report `key` and `params` in their `error` event; the app words the error again from the catalogue.

The bundled UI goes through `POST /api/ui/call` and `GET /api/dashboard`, where `localise` recognises stored English sentences in the message fields of the answer (jobs keep their error, title and progress message as text; records and parameters are never touched) and `wire` turns every coded text into `{key, params, text}`; errors on those paths carry `key`, `params` and a coded `hint`. `client/src/msgs.js` holds `[español, English]` for each (`err_<key>`, `hint_<key>`, `msg_<key>`) and `t.msg`, `t.error` and `t.hint` format them; a key the build does not know falls back to the English `text`. The REST agent route and MCP never go through `wire`: assistants get the plain sentences. `tests/test_messages.py` checks that the client has every key in both languages with the same placeholders, that every call site names an existing key with all its parameters, and that no `PygmalionError` is raised with a free sentence. Text from libraries (a Python traceback, a Hugging Face error) stays as it is.

## Time estimate

`vram.estimate_time` is a throughput model, not a benchmark: `tokens = records visited x average record (at most seq_len)`, `throughput = 1600 / billions of parameters x card speed x 1.6 for LoRA` (at most 8 000 tokens/s), `time = start-up + tokens x 1.15 / throughput + steps x 0.4 s`. The plan stores the result on the job (`params._estimate`); `JobManager.timing` shows it until the worker's measured `avg_tokens_per_s` and `eta_s` exist (from the first step), then shows those, and when the job is done the elapsed time next to the estimate.
