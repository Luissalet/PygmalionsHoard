# Pygmalion's Hoard

## Unified training and evaluation studio

Pygmalion incorporates every Galton capability natively in one ASGI process on port 5202. The **Functions** page has independent Training, Evaluation and Automatic evaluation switches, all OFF on first use. Static MCP discovery never starts a server or engine. Start the app explicitly from Hoard Hub, then enable the function you want.

Training retains datasets, LoRA/QLoRA, merges, context extension, conversion, quantization, capture, lineage and publishing. Evaluation retains model discovery, suites and benchmark imports, runs/resume/results, statistics/comparison, arena, routing, rule testing, simulated users, harness matrices and agent campaigns. Training evaluations call the incorporated evaluator directly in process. Periodic discovery, watch and housekeeping additionally require the Automatic evaluation switch; saved settings remain unchanged. Opening a module never resumes historical queued jobs or campaigns. Resume them explicitly when wanted.

Native pages live at `/training/` and `/evaluation/`, without frames. Existing `/capture/...` endpoints remain available. The combined MCP catalogue has **117 tools**: 50 training, 65 evaluation and `pygmalion_modules` / `pygmalion_set_module`. Three collisions use `evaluation_settings_get`, `evaluation_settings_set` and `evaluation_gpu_status`.

Data stays in its original location through `data/modules.json` (`data_paths.training`, `data_paths.evaluation`). No database, dataset, model, result or lineage copy/conversion is required. `python scripts/adopt_evaluation.py --evaluation-data <folder>` configures the existing folder with all modules OFF. Disabling drains workers and requests before database close; saved data remains.

GPU leases and the family's configured global model service remain in force. Product verification uses invented data and fake workers, without certifying a new live-model result. See [fusion architecture](docs/FUSION.md), [evaluation API](docs/evaluation/API.md) and [source provenance](provenance/galton/source.json).

### Evaluation tools (65):

- `galton_overview` — Models, routing table, running run, notices, GPUs and next steps. Estado general del banco de pruebas.
- `galton_status` — Health: scheduler, watch, llama-server, GPUs, settings. Estado de Galton.
- `models_list` — List models (servers, Ollama, GGUF files) with measured/stale flags. Lista de modelos.
- `models_refresh` — Rediscover models: Ollama, llama-server ports, GGUF folders. Buscar modelos nuevos.
- `model_get` — One model: aliases, metadata, where it would run, scores, speed, memory. Detalle de un modelo.
- `model_add` — Register a server URL or a GGUF file to measure. Añadir un modelo.
- `model_update` — Enable or disable a model, rename it, add aliases, allow a remote endpoint. Editar un modelo.
- `model_remove` — Remove a model and its stored results (confirm=true). Quitar un modelo.
- `suites_list` — List test suites with category and case count. Lista de suites de pruebas.
- `suite_get` — One suite with its cases (prompt preview, checker, weight). Detalle de una suite.
- `case_get` — One case in full: prompt, tools, checker, reference answer, result count. Detalle de un caso.
- `suite_create` — Create your own suite, optionally with cases. Crear una suite propia.
- `suite_update` — Rename or change the description, category or answer length of your suite. Editar suite.
- `suite_duplicate` — Copy any suite (also a built-in one) into an editable suite of your own. Duplicar una suite.
- `suite_remove` — Delete one of your suites and its cases (confirm=true). Borrar suite propia.
- `case_add` — Add a case (prompt + expected answer or checker) to your suite. Añadir un caso de prueba.
- `case_update` — Change a case of your suite: prompt, expected answer, checker, weight. Editar un caso.
- `case_remove` — Delete a case of your suite (confirm=true). Borrar un caso.
- `cases_import` — Import cases from JSONL or CSV text or an absolute path. Importar casos.
- `benchmark_import` — Import IFMTBench (instruction-following translation) as your suite. Importar benchmark de traducción.
- `case_try` — Run one case (saved or draft) on one model and show answer + checker detail. Probar un caso.
- `run_plan` — Preview a run: cases per model, memory, where each would run, problems. Plan de una ejecución.
- `run_start` — Start a run: suites x models (ids, names or specs of a file/server). Lanzar una medición.
- `run_status` — Progress of a run: per model state, cases done, memory, errors. Estado de una ejecución.
- `run_cancel` — Cancel a run: the request in flight is dropped, results kept, shared servers untouched. Cancelar ejecución.
- `run_discard` — Discard a run's results from all statistics and routes (confirm=true). Descartar una ejecución.
- `run_restore` — Undo run_discard: the results of the run count again. Restaurar una ejecución descartada.
- `run_resume` — Continue an interrupted run, asking only what it did not measure. Continuar donde se quedó.
- `runs_list` — Recent runs with state and progress. Historial de ejecuciones.
- `run_results` — Per-case results of a run: answer, score, checker detail, timing. Resultados de una ejecución.
- `galton_run` — Measure these models now (quick suite by default). Medir estos modelos ahora.
- `measure_new` — Run the quick suite on every model that is new or changed. Medir lo nuevo.
- `judge_run` — Grade the answers that waited for the judge model. Pasar el juez a lo pendiente.
- `leaderboard` — Ranking by category or suite with 95 % intervals, speed and memory. Clasificación de modelos.
- `compare` — Paired comparison of two models: wins/losses, McNemar, bootstrap, verdict. Comparar dos modelos.
- `gates_correlation` — Do the checks of a case fail together? phi, lift, kappa, redundant gates. Correlación entre comprobaciones.
- `recommend` — Best measured model for a task described in words, with why and runner-up. Qué modelo usar.
- `routes_get` — The routing table: winner per task, what changes if published now. Tabla de rutas.
- `routes_publish` — Publish the routing table to routes.json for Hoard Link. Publicar rutas.
- `arena_next` — A blind pair of answers to the same prompt to vote on. Siguiente par de la arena.
- `arena_vote` — Vote a|b|tie|both_bad on an arena pair. Votar en la arena.
- `arena_ratings` — Bradley-Terry ratings from the arena votes, per category. Puntuaciones de la arena.
- `evaluation_settings_get` — All settings with values, types and defaults, plus the GPU state. Ver ajustes.
- `evaluation_settings_set` — Change settings: allowed GPUs, llama-server path, judge, runner, watch, routes policy. Cambiar ajustes.
- `evaluation_gpu_status` — Per GPU: total/used/free, allowed or reserved, leases and queue. Estado de las GPU.
- `notices_list` — Notices: regressions, improvements, new or changed models, failed runs. Avisos.
- `housekeeping_run` — Tidy caches, unused images and old unvoted arena pairs; stop leftover servers. Mantenimiento.
- `rulebench_tasks` — Rule bench tasks and the exact prompts for variant A/B/C and a pressure. Tareas del banco de reglas.
- `rulebench_score` — Score an answer to a rule-bench task: hidden tests, lines beyond the reference, lost comments. Puntuar.
- `harness_catalog` — Harnesses, variants, tasks and tools of the harness bench. Catálogo de arneses y tareas.
- `harness_matrix_start` — Run a model x harness x task matrix: model or harness at fault. Lanzar matriz de arneses.
- `harness_matrix_status` — Progress of a harness matrix: per model state, trials done, errors. Estado de la matriz.
- `harness_matrix_result` — Harness matrix as a table with a short conclusion: harness vs model. Resultado de la matriz de arneses.
- `harness_cell` — One matrix cell: its tasks and, for one task, the traces and the reason of each failure. Detalle de una celda.
- `harness_matrix_cancel` — Cancel a harness matrix: results kept, shared servers untouched. Cancelar matriz de arneses.
- `harness_matrix_list` — Recent harness matrices with state and conclusion. Historial de matrices de arneses.
- `agentbench_catalog` — Agent bench: benchmarks, their task subsets and the configured agent harnesses. Catálogo del banco de agentes.
- `agentbench_start` — Run real agent harnesses on published benchmarks with one model. Lanzar campaña de agentes.
- `agentbench_status` — Progress of an agent-bench campaign. Estado de la campaña de agentes.
- `agentbench_result` — Agent-bench board: pass rate per benchmark and harness, with ranking. Resultado de la campaña.
- `agentbench_cell` — One cell of a campaign: every trial of a harness on a benchmark. Detalle de una celda.
- `agentbench_trial` — One trial with the logs the harness and the verifier left. Detalle de una prueba con sus registros.
- `agentbench_sync` — Read a campaign's job folders again into the database. Releer las carpetas de la campaña.
- `agentbench_cancel` — Cancel an agent-bench campaign: trials already finished are kept. Cancelar campaña de agentes.
- `agentbench_list` — Recent agent-bench campaigns. Historial de campañas de agentes.

A local studio to adapt language models. It prepares datasets, trains LoRA and QLoRA adapters, merges adapters and whole models, extends the context window, converts to GGUF, quantizes with an importance matrix computed on your own text, measures the result against its parent and publishes the winner to Ollama or as a llama.cpp server. Every file it produces keeps its lineage, and the recipe that made it can be exported and replayed.

Part of the Hoard family of local apps: it runs on your PC, keeps its data in `data/`, works on its own in the browser and can be driven by an assistant over MCP.

[Versión en español](README.es.md)

## What it does

- **Datasets.** Sources: loose files, a folder, pasted or file JSONL and CSV, tools of other Hoard apps (through the family hub) and synthetic examples written by a teacher model from your own texts. Operations run in order: exact and near-duplicate removal (MinHash over 5-gram shingles), length and language filters, personal-data masking or dropping (emails, phones, IBAN, cards, DNI/NIE/CIF; the identifiers are reported only when their check digit is right, and a phone when it has a country prefix, a label or the usual digit groups). The train/eval split is seeded. Versions are immutable; reviewing records (accept, reject, edit) makes a new version. Statistics: records, tokens, length histogram, role balance, languages.
- **Training.** LoRA and QLoRA (4-bit NF4, bfloat16 compute) on a causal language model, with loss only on assistant tokens for chat data, packing for raw text, periodic evaluation, checkpoints you can resume from, a cosine schedule and a fixed seed. Before starting, `train_plan` estimates the GPU memory with its formula, estimates the time and says which allowed GPU fits. The time comes from the tokens to process (the records the steps visit x the average record, cut at `seq_len`, x padding) divided by a throughput model: 1 600 tokens/s per billion parameters for QLoRA on a 16 GB 50-series card, 1.6x that for plain LoRA, scaled by the card's speed and capped for very small models, plus a start-up and a per-step overhead. The constants are rough and the plan shows the formula with its numbers. Once the job runs, the measured tokens/s and the real ETA replace the estimate (the job view says which one it shows), and a finished job shows how long it took against the estimate.
- **Merging.** An adapter into its base (tensor by tensor, without loading the whole model), or several models of the same architecture with linear, SLERP, TIES or DARE. Compatibility is checked first.
- **Context extension.** A copy of a model with YaRN rope scaling for 2x, 4x or 8x the context, a table of KV-cache memory per context length from the GGUF metadata, and the needle-in-a-haystack results per length.
- **GGUF and quantization.** Conversion with the installed llama.cpp (the architecture is checked first), importance matrix on a leased GPU, several quantization types in one go (Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, Q3_K_M, IQ3_M) and perplexity with its error. The perplexity is stored on the measured file with its error, text, context and chunks; when a pipeline quantizes to several types with perplexity on, the unquantized file and every quantization are measured the same way, and the base model's own file when it exists or is being built for the evaluation. Linaje and Cuantizar show them side by side with the change against the parent file (what the quantization cost) and against the base's file of the same type (what the fine-tune did).
- **Evaluation.** Each GGUF or Ollama result is compared through the incorporated Evaluation module in this process, and the verdict (better, worse, no clear difference) with its difference and interval is stored on the result together with the kind of reference it rests on. What it is compared with depends on what is asked. The intents that ask whether a training helped or hurt (`dataset`, `style`, `writing`, `code`, `general`, `smoke`) compare with the **base model the training started from, written like the result**: the same quantization type and, when the result used an importance matrix, a matrix computed on the same calibration text. The nearest GGUF ancestor of a fine-tuned file is the f16 file of the same fine-tune, which would measure the quantization and not the training. When the base has no file like that yet, `evaluate_start` queues the steps that make it (convert the base folder, matrix, quantize; whatever already exists is reused) as one pipeline that ends in the evaluation; the files stay in the lineage below the base model and the next evaluation finds them. `context` compares with the model before the context extension, built the same way. A plain quantization (no training behind it; intent `quant`, its default) keeps comparing with the f16 file it was made from. `against` always wins. `evaluate_plan` (and `artifact_get`, as `evaluate_plan`) says which reference an evaluation would use and whether it has to be prepared first; the Evaluate box in Linaje shows that sentence. A result whose base model is not in the lineage is compared with its parent file and the verdict carries a warning. A result that descends from a training job whose dataset holds out records is measured by default on those records (`intent: dataset`): the held-out records of that dataset version become a Galton suite `pyg-<dataset>-v<n>-eval` (built once per version, reused afterwards; each record is a case with the reference answer and a judge checker), the parent and the child answer it, and a small general suite (`rapida`) runs beside it to catch regressions. Both verdicts are stored and shown; a worse general result stops the promotion suggestion. The judge is Galton's own model (its `judge.contestant` setting): without one the evaluation stops before it starts and says where to set it. `galton.eval_cases` caps the number of cases. On the held-out records both models answer without thinking first (`effort: off`, the way the records were written) unless the evaluation passes its own settings.
- **Capture.** A chat-completions proxy in front of a llama.cpp or vLLM server, to turn the real turns of Faustus, of the Hoards or of any agent harness into training data with a loss mask. Point the harness at `http://127.0.0.1:5202/training/capture/v1` (one session), `/capture/<session>/v1` or `/capture/<harness>/<session>/v1`; the session and the harness can also be the API key (`session` or `harness:session`) or the `X-Capture-Session` and `X-Capture-Harness` headers. Each call is forwarded to the configured server without streaming (a client that asked for a stream gets the complete answer re-emitted as events, tool calls included) and the client receives the normal answer, without the extra fields; its sampling parameters are not touched unless `capture.faithful_sampling` is on, which neutralises `top_k`, `top_p` and `min_p` so the logged log-probabilities describe the distribution the token was drawn from (for reinforcement learning). Every call is stored as a node: the ids of the prompt (after the chat template), the ids and log-probabilities of what the model sampled, the messages, the tools, the sampling parameters, the model, the server and a verification state. vLLM gives the ids directly (`prompt_token_ids`, `token_ids`). llama.cpp gives the ids and log-probabilities of the sampled tokens but not those of the prompt: they are rebuilt with the server's own `/apply-template` and `/tokenize` and the count is checked against `usage.prompt_tokens`; a call that does not match (tools on a server started without `--jinja`, images, an unusual template) is stored as *unverified* and left out of the exports unless asked for. When llama.cpp counts one more generated token than it lists, it is the end-of-turn token: it is appended (its log-probability is unknown, so it is null) so the model learns to stop. A call hangs from an earlier call of its session when all the tokens of that one (prompt and answer) are the exact beginning of its prompt; the calls of a session form a prefix tree, siblings are branches (retries, alternatives, sub-agents) and a call that continues nothing starts a new root. Each root-to-leaf path is one training sequence: context has mask 0, the tokens the model sampled have mask 1, and a token shared by several paths is trained on once. Sessions take an optional `reward` and `outcome` (set later with `capture_outcome`; empty when nothing verified the result). The export (`capture_export`; `dry_run` reports the split first) filters by session, harness, date and reward, can balance by harness (`balance_by_harness`: as many sequences per harness as the smallest one has, or a cap, with a seeded sample and a report per harness) and writes, under `data/capture/exports/`, an SFT token file (`input_ids`, `labels` with -100 on context), an RL token file (loss mask, per-token log-probabilities, reward) and, if a dataset name is given, a chat dataset registered in Datasets that Entrenar can train on. Calls are stored as the part that follows their parent, so a long agent session costs its new tokens and not its whole context at every step.
- **Publishing.** A Modelfile and `ollama create` with the tag `pyg-<name>:<tag>`, or a llama.cpp server entry in the hub's backend list (never started automatically, only on allowed GPUs). Unpublishing asks for confirmation.
- **Jobs.** Two lanes, one job each: `gpu` and `cpu`. A pipeline chains steps (train, merge, convert, matrix, quantize, perplexity, publish, evaluate); if one fails the rest do not start and you can resume. Cancelling a training run saves a checkpoint first. Jobs found running after a restart are marked interrupted and can be resumed. A job's page lists every step of its pipeline, including the ones not created yet, with their current states, and keeps refreshing while the chain runs. The scheduler reads what is running from the database, closes rows that claim to run while nothing runs them, and the Trabajos badge counts only running and waiting jobs. Program output is cleaned before it is stored: terminal escape sequences are removed and spinner or progress lines that a program redraws in place (for example `ollama create`) are kept as their last state. Every job reports to the family hub with the canonical events `pygmalion.job.queued|started|progress|done|failed|cancelled` (`{job_id, title, kind, progress, gpu, eta_s, url, error}`; `kind` is `train`, `merge`, `convert`, `quantize` or `publish`, and a finished publish carries `model`, the name it was published under), which the hub's Work tab shows and a hub rule uses to ask Galton to measure a freshly published model.
- **GPUs.** Only the allowed GPUs are used (default 2 and 3), always through a lease from the family hub. GPUs 0 and 1 belong to the owner of the computer and need an explicit confirmation to be allowed.
- **Lineage.** Base, adapter, merged, context variant, GGUF, matrix and Ollama artifacts form a graph with the dataset hash, the parameters and the training summary. The recipe export lists every step from the base in order.

## Screens

Spanish by default, English with one click, dark, desktop and phone.

- **Panel**: environment status, GPUs with leases, running jobs with a loss sparkline, latest results with their verdict, quick actions.
- **Modelos base**: local models with architecture, size and what they support; Hugging Face search and download with a size confirmation.
- **Datasets**: list, builder with sources and operations, statistics, records with review and editing, versions.
- **Entrenar**: base, dataset version, method and parameters with the plan and memory estimate, steps after training, start.
- **Trabajos**: queue and history; each job with its loss curves, progress, ETA, GPU memory, log, cancel and resume.
- **Fusionar**: adapter into base; model merges with the compatibility check.
- **Cuantizar**: choose types, matrix and calibration text; table of results with size, perplexity and verdict.
- **Contexto**: build a context variant, see the memory table and the needle results per length.
- **Linaje**: graph and table of artifacts, detail with recipe export, evaluation, publishing and notes.
- **Captura**: the capture proxy (server to capture, start and stop, the addresses to paste into a harness with copy buttons), sessions with their calls, tokens, unverified count and reward, the call tree of a session (branches, new roots, verification state; a call opens in a drawer with its messages, tools, answer and the sampled tokens coloured by log-probability), the reward form, and the export with filters, balance by harness, a dry run that reports the split and the files it wrote.
- **Ajustes**: allowed GPUs, paths, defaults, teacher model, link to Galton, Hugging Face token (write-only), environment check with the command that fixes each gap.

## What it cannot do

- It does not install PyTorch or llama.cpp. The training, merge and probe workers run in a separate Python environment that you prepare (see Install); the app tells you what is missing and the command to fix it.
- It trains only decoder language models that the installed `transformers` can load as a causal model, from safetensors checkpoints. Pickle-based adapters and weights are refused.
- Training, the transformers path of the adapter merge and every GPU step have been exercised against fake workers and fake llama.cpp programs in the tests, not against a real GPU. The first real run on your machine is the real test.
- YaRN changes the configuration, not the weights: quality at the far end of the window has to be measured, and that needs the Evaluation module enabled on the Functions page.
- A publish writes `num_ctx` from the `publish.num_ctx` setting (8192 by default, never above the context the model was trained for; editable in the publish form), because Ollama reserves memory for the whole window. A context variant is published at the length it was built for.
- The held-out evaluation needs a judge model configured in Galton; without it the run is refused with the setting to change.
- Preparing the base model's reference file converts and quantizes the base model, which takes disk space and time; it is done once per base model, quantization and calibration text. A merge of whole models (without a training) has no base to be measured against and is compared with its parent file.
- Evaluation, family-app sources and synthetic data need the family hub or the other app running; without them those steps fail with a clear message and the rest keeps working.
- Synthetic examples are only as good as the teacher model; they stay pending until you approve them.
- The time estimate is a rough throughput constant (1 600 tokens/s per billion parameters for QLoRA on a 16 GB 50-series card), not a measurement of your machine; the first steps of the job replace it. Text that comes from a library (a Python error, a Hugging Face message, Galton's own warnings) is shown as it arrived, in its language.

- Capture needs a llama.cpp (`llama-server`) or vLLM server it can reach; it stores the ids the server reports and does not tokenize anything itself. On llama.cpp the prompt ids are rebuilt, not reported: tools are rendered only by a server started with `--jinja`, and a prompt with images does not match its count, so those calls are marked unverified. The link between calls is an exact token prefix: a harness that rewrites an earlier turn (drops its reasoning, trims a tool result) changes those tokens and the chain breaks there, which starts a new root. The chat dataset it registers flattens tools to text (a tool result becomes a user turn, a tool call a `<tool_call>` block) because the built-in trainer reads plain system, user and assistant turns; the token files keep the exact tokens and are for an external trainer or for reinforcement learning, they are not read by the built-in trainer. Token ids belong to the tokenizer of the server's model: train only a model with the same tokenizer on them. The proxy has no key of its own: it answers whoever reaches the app's port (loopback unless `PYGMALION_ALLOWED_HOSTS` says otherwise) and it stores every prompt it forwards, so it is off until started and the sessions can be deleted.

## Faustus panel

A «Faustus» button opens the family's shared assistant panel without leaving the app. Faustus keeps the conversations
(also listed in Faustus Studio), runs the agent with the globally selected model and uses this app's own tools. Each turn
carries the base model, dataset (and the version on screen), job or artifact open; unsaved fields of the training setup, the dataset builder and the artifact name and notes. After a turn the lists and the open view refresh in place, without discarding what you were typing. The
panel comes from Hoard Link (`install_spa`) and talks to Faustus through this app's relay, so no token reaches the browser.

## Install

Requirements: Python 3.11 or newer (tested on 3.11 and 3.13), Node 22 only to rebuild the UI.

```bash
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt      # Windows; venv/bin/python elsewhere
```

The trainer environment is separate and not installed by this file. Create it once and point `env.python` in Settings at it:

```bash
python -m venv <work>/venv
<work>/venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
<work>/venv/Scripts/python -m pip install -U transformers peft trl accelerate datasets bitsandbytes safetensors huggingface_hub gguf numpy
```

For GGUF work set `llama.bin_dir` (llama-quantize, llama-imatrix, llama-perplexity, llama-server) and `llama.src_dir` (llama.cpp sources with `convert_hf_to_gguf.py` and `convert_lora_to_gguf.py`). Settings and `env_check` show exactly what is missing; `env_check` lists every CUDA device the training environment sees (index in nvidia-smi order, name, total memory). The architecture support check reads `convert_hf_to_gguf.py` and, in newer llama.cpp sources, the whole `conversion/` package. The dashboard counts read the same source as `bases_list` and the page reloads after an environment check and after any write that changes a count.

## Run

```bash
venv/Scripts/python -m pygmalion_hoard                        # http://127.0.0.1:5202
python scripts/launch.py                                      # free port and open the browser
```

The UI is prebuilt in `pygmalion_hoard/static`. To rebuild it: `npm install && npx vite build`. A second start of the app (or of `scripts/launch.py`) while one is running does not touch anything: it says where the running copy is and exits. The launcher, the request guard, the health route, the error envelope, the installable-app files and the single-page server are the shared ones from Hoard Link.

Environment: `PYGMALION_PORT` (5202), `PYGMALION_DATA_DIR`, `PORT_STRICT=1`, `PYGMALION_SCHEDULER=0` (no background work), `PYGMALION_OFFLINE=1` (no network), `PYGMALION_WORKERS_DIR`, `PYGMALION_ALLOWED_HOSTS`, `PYGMALION_HTTP_TIMEOUT_S`, `PYGMALION_HF_TOKEN` (or save it in Settings). Everything else is a setting stored in `data/pygmalion.db`.

## Assistants (MCP)

`mcp_server.py` is a stdio MCP bridge named `pygmalion-hoard` (the combined static catalogue). It never opens the database: it proxies every call to the running app with the token in `data/mcp-token` (the token is created once and kept across restarts). When the app is unavailable it asks for an explicit start; it never starts the server. Tools that start a job take `wait_s`, at most 150 seconds: when the job is still running when the wait ends the answer says `still_running` and `job_get` follows it. `faustus-plugin.json` describes the app, its health check and the bridge for Faustus and the Hoard Hub.

Tools (50): `pygmalion_overview`, `env_check`, `bases_list`, `base_get`, `hf_search`, `base_download`, `datasets_list`, `dataset_get`, `dataset_create`, `dataset_records`, `dataset_review`, `dataset_preview_source`, `dataset_apply`, `dataset_delete`, `train_plan`, `train_start`, `merge_check`, `merge_lora_start`, `merge_models_start`, `ctx_extend_start`, `ctx_fit`, `convert_start`, `quantize_start`, `perplexity_start`, `jobs_list`, `job_get`, `job_cancel`, `job_resume`, `job_delete`, `artifacts_list`, `artifact_get`, `artifact_recipe`, `artifact_update`, `artifact_delete`, `lineage_graph`, `evaluate_plan`, `evaluate_start`, `publish_ollama`, `publish_llama`, `unpublish`, `settings_get`, `settings_set`, `secret_set`, `gpu_status`, `capture_proxy`, `capture_sessions`, `capture_session`, `capture_outcome`, `capture_export`, `capture_delete`. Arguments in [docs/API.md](docs/API.md).

Deleting needs `confirm=true`. Allowing a GPU that belongs to the owner needs `confirm_reserved=true`, which an assistant must not pass unless the owner said so.

## Data and privacy

Everything stays on the computer. `data/` holds the database, dataset versions, logs (including `logs/pygmalion-hoard.log`, a rotating log of the app itself, which is what remains when it is started without a console, for example with `pythonw`), the MCP token and `secrets.env` (the Hugging Face token, owner-only permissions, never shown back or logged). The work folder (`paths.work`, by default `data/work`) holds `hf/` (downloaded models), `outputs/`, `jobs/` and `calib/`. The only network use is Hugging Face (search and download, when you ask), the family hub and Galton on this computer. Imported files and model output are treated as untrusted: only safetensors and JSON are read from adapters, and dataset text is never executed.

## Development

```bash
python -m pytest -q                  # whole suite; no network, no GPU, no real torch
python scripts/gen_api_doc.py        # regenerate docs/API.md after changing a tool
npx vite build                       # rebuild the UI
```

Stopping the app is clean: the job lanes, the event worker and the lease helpers are stopped and joined with a timeout, and the suite checks after every test that no thread of the app is left running (`tests/test_shutdown.py`).

FastAPI + SQLite (WAL) + a two-lane job manager; React 19 + Vite + Tailwind UI. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [AGENTS.md](AGENTS.md).

## License

MIT
