# Pygmalion's Hoard

A local studio to adapt language models. It prepares datasets, trains LoRA and QLoRA adapters, merges adapters and whole models, extends the context window, converts to GGUF, quantizes with an importance matrix computed on your own text, measures the result against its parent and publishes the winner to Ollama or as a llama.cpp server. Every file it produces keeps its lineage, and the recipe that made it can be exported and replayed.

Part of the Hoard family of local apps: it runs on your PC, keeps its data in `data/`, works on its own in the browser and can be driven by an assistant over MCP.

[Versión en español](README.es.md)

## What it does

- **Datasets.** Sources: loose files, a folder, pasted or file JSONL and CSV, tools of other Hoard apps (through the family hub) and synthetic examples written by a teacher model from your own texts. Operations run in order: exact and near-duplicate removal (MinHash over 5-gram shingles), length and language filters, personal-data masking or dropping (emails, phones, IBAN, cards, DNI/NIE). The train/eval split is seeded. Versions are immutable; reviewing records (accept, reject, edit) makes a new version. Statistics: records, tokens, length histogram, role balance, languages.
- **Training.** LoRA and QLoRA (4-bit NF4, bfloat16 compute) on a causal language model, with loss only on assistant tokens for chat data, packing for raw text, periodic evaluation, checkpoints you can resume from, a cosine schedule and a fixed seed. Before starting, `train_plan` estimates the GPU memory with its formula, estimates the time and says which allowed GPU fits. The time comes from the tokens to process (the records the steps visit x the average record, cut at `seq_len`, x padding) divided by a throughput model: 1 600 tokens/s per billion parameters for QLoRA on a 16 GB 50-series card, 1.6x that for plain LoRA, scaled by the card's speed and capped for very small models, plus a start-up and a per-step overhead. The constants are rough and the plan shows the formula with its numbers. Once the job runs, the measured tokens/s and the real ETA replace the estimate (the job view says which one it shows), and a finished job shows how long it took against the estimate.
- **Merging.** An adapter into its base (tensor by tensor, without loading the whole model), or several models of the same architecture with linear, SLERP, TIES or DARE. Compatibility is checked first.
- **Context extension.** A copy of a model with YaRN rope scaling for 2x, 4x or 8x the context, a table of KV-cache memory per context length from the GGUF metadata, and the needle-in-a-haystack results per length.
- **GGUF and quantization.** Conversion with the installed llama.cpp (the architecture is checked first), importance matrix on a leased GPU, several quantization types in one go (Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, Q3_K_M, IQ3_M) and perplexity with its error. The perplexity is stored on the measured file with its error, text, context and chunks; when a pipeline quantizes to several types with perplexity on, the unquantized file and every quantization are measured the same way, and the base model's own file when it exists or is being built for the evaluation. Linaje and Cuantizar show them side by side with the change against the parent file (what the quantization cost) and against the base's file of the same type (what the fine-tune did).
- **Evaluation.** Each GGUF or Ollama result is compared through Galton's Hoard (through the hub, or directly as a fallback), and the verdict (better, worse, no clear difference) with its difference and interval is stored on the result together with the kind of reference it rests on. What it is compared with depends on what is asked. The intents that ask whether a training helped or hurt (`dataset`, `style`, `writing`, `code`, `general`, `smoke`) compare with the **base model the training started from, written like the result**: the same quantization type and, when the result used an importance matrix, a matrix computed on the same calibration text. The nearest GGUF ancestor of a fine-tuned file is the f16 file of the same fine-tune, which would measure the quantization and not the training. When the base has no file like that yet, `evaluate_start` queues the steps that make it (convert the base folder, matrix, quantize; whatever already exists is reused) as one pipeline that ends in the evaluation; the files stay in the lineage below the base model and the next evaluation finds them. `context` compares with the model before the context extension, built the same way. A plain quantization (no training behind it; intent `quant`, its default) keeps comparing with the f16 file it was made from. `against` always wins. `evaluate_plan` (and `artifact_get`, as `evaluate_plan`) says which reference an evaluation would use and whether it has to be prepared first; the Evaluate box in Linaje shows that sentence. A result whose base model is not in the lineage is compared with its parent file and the verdict carries a warning. A result that descends from a training job whose dataset holds out records is measured by default on those records (`intent: dataset`): the held-out records of that dataset version become a Galton suite `pyg-<dataset>-v<n>-eval` (built once per version, reused afterwards; each record is a case with the reference answer and a judge checker), the parent and the child answer it, and a small general suite (`rapida`) runs beside it to catch regressions. Both verdicts are stored and shown; a worse general result stops the promotion suggestion. The judge is Galton's own model (its `judge.contestant` setting): without one the evaluation stops before it starts and says where to set it. `galton.eval_cases` caps the number of cases. On the held-out records both models answer without thinking first (`effort: off`, the way the records were written) unless the evaluation passes its own settings.
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
- **Ajustes**: allowed GPUs, paths, defaults, teacher model, link to Galton, Hugging Face token (write-only), environment check with the command that fixes each gap.

## What it cannot do

- It does not install PyTorch or llama.cpp. The training, merge and probe workers run in a separate Python environment that you prepare (see Install); the app tells you what is missing and the command to fix it.
- It trains only decoder language models that the installed `transformers` can load as a causal model, from safetensors checkpoints. Pickle-based adapters and weights are refused.
- Training, the transformers path of the adapter merge and every GPU step have been exercised against fake workers and fake llama.cpp programs in the tests, not against a real GPU. The first real run on your machine is the real test.
- YaRN changes the configuration, not the weights: quality at the far end of the window has to be measured, and that needs Galton's Hoard running.
- A publish writes `num_ctx` from the `publish.num_ctx` setting (8192 by default, never above the context the model was trained for; editable in the publish form), because Ollama reserves memory for the whole window. A context variant is published at the length it was built for.
- The held-out evaluation needs a judge model configured in Galton; without it the run is refused with the setting to change.
- Preparing the base model's reference file converts and quantizes the base model, which takes disk space and time; it is done once per base model, quantization and calibration text. A merge of whole models (without a training) has no base to be measured against and is compared with its parent file.
- Evaluation, family-app sources and synthetic data need the family hub or the other app running; without them those steps fail with a clear message and the rest keeps working.
- Synthetic examples are only as good as the teacher model; they stay pending until you approve them.
- The time estimate is a rough throughput constant (1 600 tokens/s per billion parameters for QLoRA on a 16 GB 50-series card), not a measurement of your machine; the first steps of the job replace it. Text that comes from a library (a Python error, a Hugging Face message, Galton's own warnings) is shown as it arrived, in its language.

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

The UI is prebuilt in `pygmalion_hoard/static`. To rebuild it: `npm install && npx vite build`.

Environment: `PYGMALION_PORT` (5202), `PYGMALION_DATA_DIR`, `PORT_STRICT=1`, `PYGMALION_SCHEDULER=0` (no background work), `PYGMALION_OFFLINE=1` (no network), `PYGMALION_WORKERS_DIR`, `PYGMALION_ALLOWED_HOSTS`, `PYGMALION_HTTP_TIMEOUT_S`, `PYGMALION_HF_TOKEN` (or save it in Settings). Everything else is a setting stored in `data/pygmalion.db`.

## Assistants (MCP)

`mcp_server.py` is a stdio MCP bridge named `pygmalion-hoard`. It never opens the database: it proxies every call to the running app with the token in `data/mcp-token`, and starts the app when it is not answering. `faustus-plugin.json` describes the app, its health check and the bridge for Faustus and the Hoard Hub.

Tools (44): `pygmalion_overview`, `env_check`, `bases_list`, `base_get`, `hf_search`, `base_download`, `datasets_list`, `dataset_get`, `dataset_create`, `dataset_records`, `dataset_review`, `dataset_preview_source`, `dataset_apply`, `dataset_delete`, `train_plan`, `train_start`, `merge_check`, `merge_lora_start`, `merge_models_start`, `ctx_extend_start`, `ctx_fit`, `convert_start`, `quantize_start`, `perplexity_start`, `jobs_list`, `job_get`, `job_cancel`, `job_resume`, `job_delete`, `artifacts_list`, `artifact_get`, `artifact_recipe`, `artifact_update`, `artifact_delete`, `lineage_graph`, `evaluate_plan`, `evaluate_start`, `publish_ollama`, `publish_llama`, `unpublish`, `settings_get`, `settings_set`, `secret_set`, `gpu_status`. Arguments in [docs/API.md](docs/API.md).

Deleting needs `confirm=true`. Allowing a GPU that belongs to the owner needs `confirm_reserved=true`, which an assistant must not pass unless the owner said so.

## Data and privacy

Everything stays on the computer. `data/` holds the database, dataset versions, logs (including `logs/pygmalion.log`, a rotating log of the app itself, which is what remains when it is started without a console, for example with `pythonw`), the MCP token and `secrets.env` (the Hugging Face token, owner-only permissions, never shown back or logged). The work folder (`paths.work`, by default `data/work`) holds `hf/` (downloaded models), `outputs/`, `jobs/` and `calib/`. The only network use is Hugging Face (search and download, when you ask), the family hub and Galton on this computer. Imported files and model output are treated as untrusted: only safetensors and JSON are read from adapters, and dataset text is never executed.

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
