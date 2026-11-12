<img src="app-icon.png" width="96" alt="">

# Pygmalion's Hoard

A local studio to adapt language models (datasets, LoRA/QLoRA training, merging, quantization, publishing) with a built-in evaluation bench. [Español](README.es.md)

## What it is

Pygmalion's Hoard prepares datasets, trains LoRA and QLoRA adapters, merges adapters and whole models, extends the context window, converts to GGUF, quantizes with an importance matrix computed on your own text, measures the result against its parent and publishes the winner to Ollama or as a llama.cpp server. Every file it produces keeps its lineage, and the recipe that made it can be exported.

It also contains the whole model test bench (the former Galton's Hoard): model discovery, suites, benchmark imports, runs, statistics, arena, routing, simulated users, harness matrices and agent campaigns. Training and Evaluation are independent functions in one process on port 5202, plus an Automatic evaluation switch for periodic discovery and housekeeping. All three are off on first use; start the app from Hoard Hub, then enable what you want. Opening a module never resumes old queued jobs; resume them explicitly.

It runs on your PC, keeps its data in `data/`, works on its own in the browser and can be driven by an assistant over MCP. Pages live at `/training/` and `/evaluation/`.

## What it does

### Training
- **Datasets.** Sources: loose files, a folder, JSONL or CSV, tools of other Hoard apps through the hub, and synthetic examples written by a teacher model from your own texts (they stay pending until you approve them). Ordered operations: exact and near-duplicate removal, length and language filters, masking or dropping of personal data (emails, phones, IBAN, cards, DNI/NIE/CIF), with statistics, review, editing and versions.
- **Training.** LoRA and QLoRA (4-bit NF4) on causal language models, with loss only on assistant tokens for chat data, packing for raw text, periodic evaluation, resumable checkpoints, a cosine schedule and a fixed seed. `train_plan` estimates GPU memory and time and says which allowed GPU fits.
- **Merging.** An adapter into its base without loading the whole model, or several models of the same architecture with linear, SLERP, TIES or DARE, after a compatibility check.
- **Context extension.** A copy of a model with YaRN rope scaling for 2x, 4x or 8x the context, a KV-cache memory table per length and needle-in-a-haystack results.
- **GGUF and quantization.** Conversion with the installed llama.cpp, importance matrix on a leased GPU, several quantization types at once (Q8_0 to IQ3_M) and perplexity with its error.
- **Evaluation of results.** Each GGUF or Ollama result is compared with the base it came from, written like the result, and the verdict (better, worse, no clear difference) with its interval is stored on the result. Missing reference files are built as part of the same pipeline.
- **Capture.** A chat-completions proxy in front of a llama.cpp or vLLM server that turns real turns of Faustus, the Hoards or any agent harness into training data with a loss mask, token ids and log-probabilities, a call tree per session, rewards and filtered exports.
- **Publishing.** A Modelfile and `ollama create` with the tag `pyg-<name>:<tag>`, or a llama.cpp server entry in the hub's backend list (never started automatically). Unpublishing asks for confirmation.
- **Jobs.** Two lanes, `gpu` and `cpu`. Pipelines chain train, merge, convert, matrix, quantize, perplexity, publish and evaluate; a failure stops the rest and the pipeline can be resumed. Cancelling a training saves a checkpoint; jobs found running after a restart are marked interrupted.
- **GPUs.** Only allowed GPUs are used, always through a lease from the family hub; GPUs reserved for the computer's owner need an explicit confirmation.
- **Lineage.** Base, adapter, merged, context variant, GGUF, matrix and Ollama artifacts form a graph with dataset hash, parameters and training summary.

### Evaluation
- Model discovery (servers, Ollama, GGUF files), suites and benchmark imports, runs with resume, results, statistics and comparison, a results export to CSV or JSON.
- Arena, routing, rule testing, simulated users, harness matrices (also along a Spanish/English language axis), a code-made video bank and agent campaigns.
- Answer checks outside a run: faithfulness (claim-by-claim against the context), answer relevance and similarity to a reference (ROUGE-L, word overlap or character ratio).
- RAG pipeline evaluation: measures the retriever (recall@k, MRR, nDCG), the order and the cut, and the generator (faithfulness, coverage, abstention), and attributes each failed case to the piece that lost it. It evaluates a built-in pipeline, the document RAG of a running Faustus or any pipeline behind an HTTP request.
- Suit evaluation for Darwin's Hoard: cases with executable acceptance checks, three arms (generic tools, current suit, candidate), spread-aware aggregates, and results sent to Darwin's Hoard.
- Training evaluations call the evaluator directly in process. Evaluation tools are listed below and in [docs/evaluation/API.md](docs/evaluation/API.md).

### Screens
Spanish by default, English with one click, dark mode, desktop and phone. Pages: Panel, Base models (with Hugging Face search and download), Datasets, Train, Jobs, Merge, Quantize, Context, Lineage, Capture and Settings, plus the Evaluation pages.

### Limits
- It does not install PyTorch or llama.cpp. The training, merge and probe workers run in a separate Python environment you prepare; the app tells you what is missing and the command to fix it.
- It trains only decoder language models from safetensors checkpoints that the installed `transformers` can load; pickle-based weights are refused.
- YaRN changes the configuration, not the weights; quality at the far end of the window has to be measured with the Evaluation module.
- The held-out evaluation needs a judge model configured; without it the run is refused with the setting to change.
- Evaluation, family-app sources and synthetic data need the hub or the other app running; without them those steps fail with a clear message and the rest keeps working.
- The time estimate is a rough throughput constant, replaced by measurements as the job runs.
- Capture needs a llama.cpp (`llama-server`) or vLLM server it can reach; on llama.cpp, tools are rendered only by a server started with `--jinja`, and calls it cannot verify are marked unverified.

## How to run it

Requires Python 3.11 or newer; Node 22 only to rebuild the UI.

```bash
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt      # Windows; venv/bin/python elsewhere
venv/Scripts/python -m pygmalion_hoard                        # http://127.0.0.1:5202
python scripts/launch.py                                      # free port and open the browser
```

The trainer environment is separate. Create it once and point `env.python` in Settings at it:

```bash
python -m venv <work>/venv
<work>/venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
<work>/venv/Scripts/python -m pip install -U transformers peft trl accelerate datasets bitsandbytes safetensors huggingface_hub gguf numpy
```

For GGUF work set `llama.bin_dir` (llama-quantize, llama-imatrix, llama-perplexity, llama-server) and `llama.src_dir` (llama.cpp sources with the conversion scripts). Settings and `env_check` show exactly what is missing.

The UI is prebuilt in `pygmalion_hoard/static`; rebuild with `npm install && npx vite build`. A second start while one is running does not touch anything: it says where the running copy is and exits.

Environment: `PYGMALION_PORT` (5202), `PYGMALION_DATA_DIR`, `PORT_STRICT=1`, `PYGMALION_SCHEDULER=0` (no background work), `PYGMALION_OFFLINE=1` (no network), `PYGMALION_WORKERS_DIR`, `PYGMALION_ALLOWED_HOSTS`, `PYGMALION_HTTP_TIMEOUT_S`, `PYGMALION_HF_TOKEN` (or save it in Settings). Everything else is a setting stored in `data/pygmalion.db`.

Everything stays on the computer. `data/` holds the database, dataset versions, logs, the MCP token and `secrets.env` (the Hugging Face token, owner-only permissions, never shown back or logged). Existing training and evaluation data folders can be adopted in place through `data/modules.json` (`python scripts/adopt_evaluation.py --evaluation-data <folder>`), with no copy or conversion. See [docs/FUSION.md](docs/FUSION.md).

## Connection with Faustus and Hoard Hub

- `faustus-plugin.json` is the manifest. `mcp_server.py` is a stdio MCP bridge named `pygmalion-hoard` exposing the combined static catalogue; it never opens the database and never starts the server: it proxies every call to the running app with the token in `data/mcp-token`. Tools that start a job take `wait_s`, at most 150 seconds.
- Deleting needs `confirm=true`. Allowing a GPU that belongs to the owner needs `confirm_reserved=true`, which an assistant must not pass unless the owner said so.
- A "Faustus" button opens the family's shared assistant panel, carrying the base model, dataset, job or artifact on screen and unsaved training fields; lists refresh after each turn.
- GPU leases and the configured global model service come from the hub. The app carries skills for connected agents (`model-finetune-and-eval`, `eval-harness`, `prompt-optimizer`, `measured-improvement`, `cross-model-review`).

Training tools (50): `pygmalion_overview`, `env_check`, `bases_list`, `base_get`, `hf_search`, `base_download`, `datasets_list`, `dataset_get`, `dataset_create`, `dataset_records`, `dataset_review`, `dataset_preview_source`, `dataset_apply`, `dataset_delete`, `train_plan`, `train_start`, `merge_check`, `merge_lora_start`, `merge_models_start`, `ctx_extend_start`, `ctx_fit`, `convert_start`, `quantize_start`, `perplexity_start`, `jobs_list`, `job_get`, `job_cancel`, `job_resume`, `job_delete`, `artifacts_list`, `artifact_get`, `artifact_recipe`, `artifact_update`, `artifact_delete`, `lineage_graph`, `evaluate_plan`, `evaluate_start`, `publish_ollama`, `publish_llama`, `unpublish`, `settings_get`, `settings_set`, `secret_set`, `gpu_status`, `capture_proxy`, `capture_sessions`, `capture_session`, `capture_outcome`, `capture_export`, `capture_delete`. Arguments in [docs/API.md](docs/API.md).

Module controls: `pygmalion_modules`, `pygmalion_set_module`. Where names collide with evaluation tools, the evaluation ones are `evaluation_settings_get`, `evaluation_settings_set` and `evaluation_gpu_status`.

<details>
<summary>Evaluation tools</summary>

`galton_overview`, `galton_status`, `models_list`, `models_refresh`, `model_get`, `model_add`, `model_update`, `model_remove`, `suites_list`, `suite_get`, `case_get`, `suite_create`, `suite_update`, `suite_duplicate`, `suite_remove`, `case_add`, `case_update`, `case_remove`, `cases_import`, `benchmark_import`, `case_try`, `checkers_list`, `answer_check`, `run_plan`, `run_start`, `run_status`, `run_cancel`, `run_discard`, `run_restore`, `run_resume`, `runs_list`, `run_results`, `run_results_export`, `galton_run`, `measure_new`, `judge_run`, `leaderboard`, `compare`, `gates_correlation`, `recommend`, `routes_get`, `routes_publish`, `arena_next`, `arena_vote`, `arena_ratings`, `settings_get`, `settings_set`, `gpu_status`, `evaluation_settings_get`, `evaluation_settings_set`, `evaluation_gpu_status`, `notices_list`, `housekeeping_run`, `rulebench_tasks`, `rulebench_score`, `harness_catalog`, `harness_matrix_start`, `harness_matrix_status`, `harness_matrix_result`, `harness_cell`, `harness_matrix_cancel`, `harness_matrix_list`, `langbench_task`, `langbench_judge`, `videobench_catalog`, `videobench_prompt`, `videobench_start`, `videobench_status`, `videobench_result`, `videobench_item`, `videobench_cancel`, `videobench_list`, `logit_judge_run`, `logit_judge_result`, `logit_judge_list`, `logit_judge_probe`, `logit_judge_cancel`, `logit_judge_delete`, `agentbench_catalog`, `agentbench_start`, `agentbench_status`, `agentbench_result`, `agentbench_cell`, `agentbench_trial`, `agentbench_sync`, `agentbench_cancel`, `agentbench_list`, `pygmalion_suit_eval_define`, `pygmalion_suit_library`, `pygmalion_suit_eval_list`, `pygmalion_suit_eval_get`, `pygmalion_suit_eval_record`, `pygmalion_suit_eval_check`, `pygmalion_suit_eval_score`, `pygmalion_suit_eval_export`, `pygmalion_suit_eval_delete`, `rag_sets`, `rag_set_save`, `rag_set_get`, `rag_set_delete`, `rag_cases_generate`, `rag_case_save`, `rag_cases_review`, `rag_case_remove`, `rag_probe`, `rag_run_start`, `rag_run_status`, `rag_run_result`, `rag_run_list`, `rag_run_cancel`, `rag_run_delete`, `rag_compare`

</details>

## Project layout

| Path | Contents |
| --- | --- |
| `pygmalion_hoard/` | Python package: API, training, jobs, lineage, capture, static UI |
| `galton_hoard/` | Evaluation module |
| `client/`, `evaluation-client/` | Training and evaluation interfaces |
| `mcp_server.py` | MCP stdio bridge |
| `faustus-plugin.json` | Faustus plugin manifest |
| `compatibility/` | Compatibility manifests for former app identities |
| `scripts/` | Launcher, API doc generator, adoption helper |
| `docs/` | Architecture, fusion and API references |
| `provenance/` | Source inventory |
| `tests/`, `evaluation_tests/`, `fusion_tests/` | Test suites |

## Development

```bash
python -m pytest -q                  # whole suite; no network, no GPU, no real torch
python scripts/gen_api_doc.py        # regenerate docs/API.md after changing a tool
npx vite build                       # rebuild the UI
```

FastAPI, SQLite (WAL) and a two-lane job manager; React 19, Vite and Tailwind for the UI. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [AGENTS.md](AGENTS.md). Licence: MIT.
