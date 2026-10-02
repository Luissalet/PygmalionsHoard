"""Messages the UI shows, as a stable ``key`` plus parameters.

The API keeps an English sentence in every message: assistants read it as it is. The UI is Spanish first, so each message also carries a
``key`` and its ``params``, and the client formats them through its own translation table (``client/src/msgs.js``: ``err_<key>`` and
``hint_<key>`` for errors, ``msg_<key>`` for the rest). A test checks that every key defined here exists in both languages with the same
placeholders, and that no ``PygmalionError`` is raised with a free sentence by the app's own code.

Two catalogues:

* ``ERRORS``: ``key -> (message, hint)`` for ``PygmalionError``. ``PygmalionError("invalid", "no_dataset", ref="x")`` formats both from the
  keyword arguments.
* ``TEXTS``: ``key -> sentence`` for warnings, notes, plan lines, progress messages, job titles and stored job errors.

Parameters are plain values: numbers are rounded where they are shown, lists are joined into one string. Placeholders are bare names (no
format specs), so that the client can fill them in without Python.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

KEY = re.compile(r"^[a-z][a-z0-9_]*$")

# ------------------------------------------------------------------------------------------------- errors: key -> (message, hint)
ERRORS: dict[str, tuple[str, str]] = {
    # -- settings
    "not_gpu_index": ("'{part}' is not a GPU index.", ""),
    "setting_unknown": ("Unknown setting {key}.", "Known: {options}."),
    "setting_choice": ("{key} must be one of {options}.", ""),
    "setting_bool": ("{key} must be on or off.", ""),
    "setting_number": ("{key} must be a number.", ""),
    "setting_range": ("{key} must be between {low} and {high}.", ""),
    "gpu_reserved": ("GPU {gpus} belongs to the owner of this computer.",
        "Repeat the call with confirm_reserved=true only if the owner explicitly allows it."),
    "no_gpu_allowed_setting": ("At least one GPU must be allowed.", "Set gpus.allowed to e.g. 2,3."),
    # -- store
    "dataset_exists": ("A dataset called '{name}' already exists.", "Pick another name or add a version to it."),
    "no_dataset": ("No dataset {ref}.", "List datasets with datasets_list."),
    "no_dataset_version": ("No dataset version {ref}.", "Dataset versions are listed by dataset_get."),
    "dataset_no_version": ("Dataset {name} has no version {n}.", ""),
    "dataset_no_versions": ("Dataset {name} has no versions yet.", ""),
    "artifact_kind_unknown": ("Unknown artifact kind {kind}.", "Kinds: {options}."),
    "no_artifact": ("No artifact {ref}.", "List artifacts with artifacts_list."),
    "no_job": ("No job {id}.", "List jobs with jobs_list."),
    # -- jobs
    "env_not_set": ("The trainer environment is not set up.",
        "Set env.python in Settings (run env_check for the steps)."),
    "worker_missing": ("The worker {name} is missing: {script}.", "Reinstall the app."),
    "worker_exit": ("{name} exited with code {code}: {tail}", ""),
    "pipeline_ref_missing": ("The pipeline step needs ${name}, which no earlier step produced.", "Available: {options}."),
    "pipeline_ref_none": ("The pipeline step needs ${name}, which no earlier step produced.",
        "No earlier step has produced anything yet."),
    "job_kind_unknown": ("Unknown job kind '{kind}'.", "Kinds: {options}."),
    "job_kind_unknown_step": ("Unknown job kind '{kind}' in the pipeline.", "Kinds: {options}."),
    "pipeline_empty": ("A pipeline needs at least one step.", ""),
    "job_not_cancellable": ("Job {id} is {state}: nothing to cancel.", ""),
    "job_not_resumable": ("Job {id} is {state}: only interrupted, failed or cancelled jobs resume.", ""),
    "job_still_active": ("Job {id} is {state}: cancel it first.", ""),
    "job_no_runner": ("No runner for {kind}.", ""),
    "cancelled": ("Cancelled.", ""),
    "job_crashed": ("Unexpected error: {detail}", "See the job log."),
    # -- ops
    "repo_no_safetensors": ("{repo} has no .safetensors weights.",
        "Only safetensors checkpoints can be trained or converted."),
    "repo_too_big": ("{repo} needs {need_gb} GB and only {free_gb} GB are free.", "Free space or change paths.work."),
    "train_no_fit": ("{reason}", "Lower seq_len or batch, use QLoRA, or pass force=true to queue it anyway."),
    "models_unreadable": ("{detail}", "Each model needs .safetensors files."),
    "models_incompatible": ("The models are not compatible: {problems}",
        "Only models with the same architecture and sizes can be merged."),
    "ctx_factor_needed": ("Give a factor above 1 (2, 4 or 8) or a target length above the model's own.",
        "The model's context is {original} tokens."),
    "ctx_factor_needed_unknown": ("Give a factor above 1 (2, 4 or 8) or a target length above the model's own.",
        "The model's context is unknown."),
    "quantize_needs_source": ("Give gguf (a GGUF artifact) or model (a Hugging Face folder to convert first).", ""),
    "config_unreadable": ("{path}: config.json is missing or unreadable.", "Pick a Hugging Face model folder."),
    # -- gpus
    "gpu_no_fit": ("{reason}", "Use a smaller model, a shorter sequence, QLoRA, or allow more GPUs in Settings."),
    "no_gpu_allowed": ("No GPU is allowed.", "Set gpus.allowed in Settings."),
    "gpu_index_not_allowed": ("GPU {index} is not in the allowed list.", "Allowed GPUs are set in Settings."),
    "gpu_hub_placed": ("The hub placed the job on GPU {gpu}, which is not allowed.", ""),
    "cancelled_waiting_gpu": ("Cancelled while waiting for a GPU.", ""),
    "gpu_lease_failed": ("The GPU lease failed: {detail}", "Check the family hub (Hoard Hub) and the GPU memory."),
    # -- planner
    "plan_no_config": ("{name}: config.json has no hidden size or layers, so memory cannot be estimated.",
        "Pick a Hugging Face decoder model folder."),
    "train_method_invalid": ("method must be qlora or lora.", ""),
    # -- publish
    "ollama_tag_invalid": ("'{tag}' is not a valid Ollama tag.", "Use letters, digits, dots and hyphens."),
    "modelfile_parameter": ("Invalid Modelfile parameter '{key}'.", ""),
    "gguf_file_missing": ("The file of {name} is missing: {path}.", "Rebuild it or pick another GGUF."),
    "ollama_missing": ("Ollama was not found.", "Install Ollama or set publish.ollama_exe in Settings."),
    "ollama_create_failed": ("ollama create failed: {detail}", "Check that the Ollama service is running."),
    "no_free_port": ("No free port for the llama.cpp server.", "Raise publish.llama_port_start in Settings."),
    "llama_server_missing": ("llama-server was not found.", "Set llama.bin_dir in Settings."),
    "gpu_not_in_allowed": ("GPU {gpu} is not in the allowed list {allowed}.",
        "Change gpus.allowed in Settings if the owner agrees."),
    "backend_id_invalid": ("'{id}' is not a valid backend id.", ""),
    "backends_refused": ("backends.json refused the entry: {detail}", ""),
    "not_published": ("{name} is not published.", "Nothing to remove."),
    "only_pyg_tags": ("Only tags created by Pygmalion (pyg-...) are removed.", ""),
    "ollama_rm_failed": ("ollama rm failed: {detail}", "Check that the Ollama service is running."),
    "unpublish_nothing": ("Nothing matched what you asked to remove.", "Published entries: {entries}."),
    # -- runners
    "disk_low": ("Not enough free disk space for {what}: need about {need_gb} GB, {free_gb} GB free.",
        "Free space in the work folder or change paths.work in Settings."),
    "tool_not_found": ("{name} was not found.", "Set {setting} in Settings (env_check lists what is missing)."),
    "model_folder_missing": ("The model folder of {name} is missing: {path}.", "Download or rebuild it."),
    "dataset_no_train": ("The dataset version has no training records.",
        "Review it: accept some records or rebuild with more material."),
    "convert_unsupported": ("{note}", "Update the llama.cpp sources (llama.src_dir) or pick another model."),
    "convert_failed": ("The GGUF conversion failed: {detail}",
        "Read the job log; an unsupported architecture needs a newer llama.cpp."),
    "convert_failed_code": ("The GGUF conversion failed: exit code {code}",
        "Read the job log; an unsupported architecture needs a newer llama.cpp."),
    "imatrix_failed": ("llama-imatrix failed: {detail}",
        "If it ran out of memory, free the GPU or use a smaller model."),
    "imatrix_failed_code": ("llama-imatrix failed: exit code {code}",
        "If it ran out of memory, free the GPU or use a smaller model."),
    "quantize_failed": ("llama-quantize failed: {detail}", "Read the job log."),
    "quantize_failed_code": ("llama-quantize failed: exit code {code}", "Read the job log."),
    "perplexity_failed": ("llama-perplexity gave no result: {detail}", "Read the job log."),
    "ppl_not_stored": ("The perplexity of {name} was measured but could not be saved on the artifact.", "Resume the job to measure it again."),
    "perplexity_failed_code": ("llama-perplexity gave no result: exit code {code}", "Read the job log."),
    # -- evaluate
    "intent_unknown": ("Unknown intent '{intent}'.", "Use one of: {options} or pass suites."),
    "evaluate_kind": ("{name} is a {kind}: only GGUF files and Ollama tags can be evaluated.",
        "Convert it to GGUF first (convert_start)."),
    "galton_refused": ("Galton refused {tool}: {detail}", "Check the suites and contestants."),
    "galton_offline": ("Offline mode: Galton cannot be reached.", ""),
    "galton_token_missing": ("Galton's token was not found.", "Set galton.token_file in Settings to Galton's data/mcp-token."),
    "galton_unreachable": ("Cannot reach Galton's Hoard: {detail}.", "Start Galton's Hoard or the family hub."),
    "galton_status": ("Galton answered {status}: {detail}", "Check the arguments and Galton's token."),
    "no_comparable": ("{name} has no GGUF or Ollama ancestor to compare with.",
        "Pass against= with the GGUF or Ollama tag of the parent model."),
    "galton_no_run": ("Galton started no run (no run id in its answer).", "Answer: {answer}"),
    "galton_run_ended": ("Galton's run {run} ended as {state}: {detail}", "Open the run in Galton's Hoard."),
    "galton_run_timeout": ("Galton's run {run} did not finish in time.",
        "Raise galton.timeout_s or check the run in Galton's Hoard."),
    "galton_run_models": ("Galton's run {run} lists {n} models instead of 2.", "Open the run in Galton's Hoard."),
    "galton_parent_failed": ("Galton could not run the parent {name} in run {run}: {detail}",
        "Open the run in Galton's Hoard; free a GPU or fix the file and evaluate again."),
    "galton_child_failed": ("Galton could not run the child {name} in run {run}: {detail}",
        "Open the run in Galton's Hoard; free a GPU or fix the file and evaluate again."),
    "galton_judge_missing": ("Galton has no judge model, and the held-out records are graded by one.",
        "Choose a judge model in Galton's Hoard settings (judge.contestant) and evaluate again."),
    "galton_judge_unavailable": ("Galton's judge model could not grade the answers of run {run} ({n} still wait).",
        "Check the judge model in Galton's Hoard (setting judge.contestant); once it works, run judge_run there and evaluate again."),
    "galton_suite_empty": ("Galton imported no case into the suite {name}: {detail}", "Open the suite in Galton's Hoard."),
    "eval_no_records": ("{name} comes from no dataset version with held-out records.",
        "Evaluate with another intent (style, code, general...) or pass suites."),
    "eval_reference_missing": ("The base model {model} at {quant} does not exist, so there is nothing to compare with.",
        "Evaluate through evaluate_start, which prepares it first, or pass against= with a file to compare with."),
    # -- misc
    "gguf_truncated": ("The file ends in the middle of the header.", "The file is not a readable GGUF (v2 or v3)."),
    "gguf_string_long": ("A string in the header is implausibly long.", "The file is not a readable GGUF (v2 or v3)."),
    "gguf_value_type": ("Unknown value type {kind} in the header.", "The file is not a readable GGUF (v2 or v3)."),
    "gguf_magic": ("The magic bytes are not GGUF.", "The file is not a readable GGUF (v2 or v3)."),
    "gguf_version": ("GGUF version {version} is not supported (2 and 3 are).",
        "The file is not a readable GGUF (v2 or v3)."),
    "gguf_count": ("The key/value count is implausible.", "The file is not a readable GGUF (v2 or v3)."),
    "file_unreadable": ("Cannot open {path}: {detail}.", ""),
    "repo_id_invalid": ("'{repo}' is not a Hugging Face repository id.",
        "Use the form organisation/name, e.g. Qwen/Qwen3-4B."),
    "offline_mode": ("The app is running in offline mode.", "Unset PYGMALION_OFFLINE to search or download."),
    "hf_unreachable": ("Cannot reach Hugging Face: {detail}.", "Check the connection; local models still work."),
    "hf_not_found": ("That repository does not exist on Hugging Face (or it is private).", "Check the id."),
    "hf_refused": ("Hugging Face refused the request (gated or private repository).",
        "Accept the licence on the website and save your token in Settings."),
    "hf_status": ("Hugging Face answered {status}.", "Try again later."),
    "hf_unreadable": ("Hugging Face sent an unreadable answer.", ""),
    "hf_pipeline_unknown": ("Unknown pipeline '{pipeline}'.", "Use one of: {options}."),
    "hf_sort_unknown": ("Unknown sort '{sort}'.", "Use one of: {options}."),
    "quant_type_unknown": ("Unknown quantization type '{type}'.", "Offered: {options}."),
    "quant_type_needed": ("Pick at least one quantization type.", "Offered: {options}."),
    "outtype_unknown": ("Unknown output type '{outtype}'.", "Use one of: {options}."),
    "calib_missing": ("The bundled calibration text is missing.",
        "Reinstall the app or pass a dataset as the calibration source."),
    "not_model_folder": ("{folder} is not a Hugging Face model folder.", "It needs a config.json and .safetensors files."),
    "artifact_ref_needed": ("An artifact id, name or path is needed.", ""),
    "artifact_wrong_kind": ("{name} is a {kind}, not one of: {options}.",
        "Pick an artifact of the right kind in the lineage table."),
    "name_empty": ("The name cannot be empty.", ""),
    "artifact_published": ("{name} is published ({where}).", "Unpublish it first."),
    "files_outside_work": ("{path} is outside the work folder: its files are not deleted.",
        "Delete them by hand if you are sure."),
    "delete_failed": ("Cannot delete {path}: {detail}", "Close whatever uses it and try again."),
    # -- datasets
    "dataset_file_missing": ("The file of version {n} is missing: {path}.", "Restore data/datasets or rebuild the dataset."),
    "source_needed": ("A dataset needs at least one source.",
        "Add files, a folder, JSONL, CSV, a family app or a synthetic source."),
    "dataset_kind_unknown": ("Unknown dataset kind {kind}.", "Kinds: {options}."),
    "no_usable_records": ("No usable records came out of the sources.", "Check the format of the input."),
    "no_usable_records_notes": ("No usable records came out of the sources: {note}", "Check the format of the input."),
    "ops_removed_all": ("The operations removed every record.", "Loosen the filters."),
    "dataset_kind_mismatch": ("Dataset {name} holds {have} records, not {kind}.", "Create a new dataset for the other kind."),
    "dataset_name_needed": ("A new dataset needs a `name` (letters, digits, spaces, dots and hyphens).",
        "Or pass `dataset` to add a version to an existing one."),
    "record_out_of_range": ("{label}: record index {index} is out of range (0 to {last}).", ""),
    "record_edit_invalid": ("The edited record {index} is not valid {kind} data.", ""),
    "file_forbidden": ("Cannot read {path}: {reason}.", "Pick a file in your own folders."),
    "file_too_big": ("{name} is larger than {mb} MB.", "Split it first."),
    "pasted_too_large": ("The pasted text is too large.", "Save it to a file and use its path."),
    "source_text_or_path": ("The {type} source needs `text` or `path`.", ""),
    "csv_columns_needed": ("A CSV source needs `columns`, e.g. {{\"prompt\": \"pregunta\", \"response\": \"respuesta\"}} or {{\"text\": \"texto\"}}.",
        ""),
    "csv_column_missing": ("The CSV has no column {missing}.", "Its columns are: {columns}."),
    "files_source_needs": ("A files source needs `items` (name and text) or `paths`.", ""),
    "folder_forbidden": ("Cannot read {path}: {reason}.", "Pick a folder inside your own documents."),
    "family_offline": ("The family hub is not available.", "Start Hoard Hub, or use another source."),
    "family_source_needs": ("A family source needs `app` and `tool`.", ""),
    "family_no_answer": ("{app}.{tool} did not answer: {why}.",
        "Check that the app is running and the tool name is right."),
    "family_mapping_needed": ("A family source needs `mapping`, e.g. {{\"prompt\": \"front\", \"response\": \"back\"}}.", ""),
    "teacher_missing": ("No teacher model is available.",
        "Start a model server (Faustus, llama.cpp or Ollama) or set teacher.model."),
    "synthetic_task_needed": ("A synthetic source needs a `task`, e.g. \"genera 3 preguntas y respuestas sobre este fragmento\".",
        ""),
    "source_type_unknown": ("Unknown source type {type}.", "Use one of: {options}."),
    "unknown_secret": ("Unknown secret {name}.", "Known: {options}."),
    "confirm_permanent": ("{what} is permanent.", "Repeat the call with confirm=true if the user asked for it."),
    "convert_target_needed": ("Give model (a Hugging Face model) or adapter (a LoRA adapter).", ""),
    "env_python_unset": ("env.python is not set.", "Create the trainer environment and set env.python in Settings."),
    "env_python_missing": ("{python} does not exist.", "Check env.python in Settings."),
    "work_unwritable": ("Cannot write in the work folder: {detail}", "Check paths.work in Settings."),
    "probe_failed": ("The probe failed: {detail}", "Run the interpreter by hand to see why."),
    "probe_silent": ("The probe printed nothing.", "Run the interpreter by hand to see why."),
    # -- workers
    "args_unreadable": ("Cannot read the arguments file {path}: {detail}", ""),
    "module_unloadable": ("Cannot load {path}.", ""),
    "ctx_factor_low": ("The factor must be above 1.", "Use 2, 4 or 8."),
    "ctx_no_max_position": ("The model's config has no max_position_embeddings.", "Pass original_max_position explicitly."),
    "hf_read_failed": ("Cannot read {repo} on Hugging Face: {detail}", "Check the repository id and your connection."),
    "download_stopped": ("Stopped by request; start the download again to resume.", ""),
    "download_failed": ("Download failed: {detail}", "Run it again: the download resumes."),
    "download_no_config": ("The download finished without a config.json.",
        "The repository may not be a transformers model."),
    "adapter_config_unreadable": ("{path}: adapter_config.json is missing or unreadable.",
        "Pick the adapter folder written by a training job."),
    "merge_stopped": ("Stopped by request; start the merge again (nothing partial is kept).", ""),
    "adapter_shape_misfit": ("{name}: the adapter shapes {lora_b} x {lora_a} do not fit the base weight {weight}.",
        "The adapter was trained on a different base model."),
    "merge_cuda_missing": ("CUDA was requested for the merge but is not available.", "Use device=cpu."),
    "merge_base_invalid": ("{path}: not a Hugging Face model folder (config.json missing).", ""),
    "adapter_weights_missing": ("{path}: adapter_model.safetensors not found.",
        "Only safetensors adapters are merged; a pickle-based adapter_model.bin is refused."),
    "adapter_not_streamable": ("This adapter cannot be merged by streaming.", "Use mode=auto or transformers."),
    "merge_method_unknown": ("Unknown merge method {method}.", "Use one of: {options}."),
    "merge_slerp_two": ("slerp merges exactly two models.", "Pick two models or use linear."),
    "merge_needs_base": ("{method} needs a base model.", "Pick the model the others were tuned from."),
    "merge_needs_two": ("A merge needs at least two models.", ""),
    "train_no_cuda": ("No CUDA device is visible to the trainer.",
        "Run env_check; the job needs an allowed GPU with a working PyTorch build."),
    "bnb_missing": ("bitsandbytes is not installed in the trainer environment.",
        "pip install bitsandbytes (a CUDA build)."),
    "no_target_layers": ("No linear layers matched the requested target modules.",
        "Use target_modules=['all'] or names such as q_proj, v_proj."),
    "no_label_tokens": ("The training split produced no example with labelled tokens.",
        "Check the dataset format and that the chat template accepts its roles."),
    # -- extra
    "unknown_tool": ("Unknown tool: {name}", ""),
}

# ------------------------------------------------------------------------------------------------- texts: key -> sentence
TEXTS: dict[str, str] = {
    # -- jobs
    "job_cancelled_early": "Cancelled before it started.",
    "next_step_failed": "The next pipeline step could not start: {reason}",
    "job_lost": "The job stopped without reporting; resume it to continue.",
    # -- ops
    "download_confirm": "Repeat with confirm=true to download {gb} GB into {folder}.",
    "title_download": "download {name}",
    "title_dataset": "dataset {name}",
    "title_train": "train",
    "title_merge": "merge",
    "title_merge_models": "merge models",
    "title_ctx_extend": "extend context",
    "title_convert_adapter": "convert adapter",
    "title_convert": "convert",
    "title_perplexity": "perplexity {name}",
    "title_evaluate": "evaluate {name}",
    "title_publish": "publish {name}",
    "ctx_fit_note": "KV cache in f16; add the file size and 600 MB headroom. Cache quantization (q8_0) halves the KV column.",
    # -- vram
    "mem_formula_qlora": "weights ({params_b}B - {kept_b}B embeddings) x {q_bytes} B + {kept_b}B x {l_bytes} B = {weights_mb} MB; LoRA {lora_m}M x {state_bytes} B = {lora_mb} MB; checkpoints {batch}x{seq_len}x{hidden}x{layers}x2x{k} B = {checkpoints_mb} MB; working set {working_mb} MB; logits {batch}x{seq_len}x{vocab}x4x2 B = {logits_mb} MB; overhead {fixed_mb} MB + {pct} % = {overhead_mb} MB",
    "mem_formula_lora": "weights {params_b}B x {l_bytes} B = {weights_mb} MB; LoRA {lora_m}M x {state_bytes} B = {lora_mb} MB; checkpoints {batch}x{seq_len}x{hidden}x{layers}x2x{k} B = {checkpoints_mb} MB; working set {working_mb} MB; logits {batch}x{seq_len}x{vocab}x4x2 B = {logits_mb} MB; overhead {fixed_mb} MB + {pct} % = {overhead_mb} MB",
    "pick_no_inventory": "No GPU inventory (nvidia-smi not found).",
    "pick_no_fit": "{total_mb} MB do not fit in the allowed GPUs together ({capacity_mb} MB).",
    "time_note": "Rough estimate from a throughput constant; once the job runs, the measured speed and the real ETA replace it.",
    "time_formula": "start-up {startup_s} s + {tokens} tokens x {padding} (padding) / {tok_s} tokens/s + {steps} steps x {step_s} s. Throughput = {const} / {params_b}B params x {scale} (card) x {speed} ({method}), at most {cap} tokens/s.",
    # -- gpus
    "gpu_status_none": "No NVIDIA GPU was found (nvidia-smi is not available).",
    "gpu_plan_wait": "The estimate ({total_mb} MB) is more than the free memory of every allowed GPU right now; the job will wait for the family hub to free one.",
    "gpu_plan_no_fit": "The estimate ({total_mb} MB) does not fit in the allowed GPUs.",
    "gpu_plan_no_inventory": "No GPU inventory: the estimate cannot be compared with the free memory.",
    # -- planner
    "plan_seq_lowered": "seq_len lowered to {seq_len} so the run fits {mb} MB.",
    "plan_lora_big": "Plain LoRA keeps the model in bf16 and does not fit the allowed GPUs; QLoRA needs about a quarter of the memory.",
    "plan_few_steps": "Fewer than 20 optimizer steps: the adapter will hardly change the model. Add data or lower grad_accum.",
    "plan_few_records": "Only {n} training records: expect overfitting; 200 or more is a safer start.",
    "plan_no_inventory": "No GPU inventory: nvidia-smi was not found, so the fit cannot be checked.",
    "plan_no_fit": "The estimate does not fit the allowed GPUs.",
    "plan_wait_free": "The estimate does not fit the free memory of any allowed GPU right now; the job will wait for the hub to free one.",
    "title_imatrix": "importance matrix",
    "title_quantize": "quantize {type}",
    "title_perplexity_step": "perplexity",
    "title_perplexity_of": "perplexity ({what})",
    "title_baseline_perplexity": "baseline: perplexity ({what})",
    "title_publish_step": "publish",
    "title_evaluate_step": "evaluate",
    "title_baseline_convert": "baseline: convert",
    "title_baseline_imatrix": "baseline: importance matrix",
    "title_baseline_quantize": "baseline: quantize {type}",
    # -- publish
    "publish_note_rope": "The GGUF carries {type} rope scaling (factor {factor}); Ollama reads it from the file.",
    "publish_note_no_template": "The GGUF has no chat template: pass template= or Ollama will use a plain prompt.",
    "publish_note_llama": "Added to the hub's backends. Start it from the hub; Pygmalion does not start it.",
    "publish_note_ctx_above": "num_ctx {ctx} is above the context the model was trained for ({trained}).",
    "note_unpublished": "Unpublished.",
    # -- runners
    "what_merged_model": "the merged model",
    "what_gguf_file": "the GGUF file",
    "what_quantized_file": "the quantized file",
    "progress_training": "training",
    "progress_convert_adapter": "converting the LoRA adapter to GGUF ({outtype})",
    "progress_convert_model": "converting the model to GGUF ({outtype})",
    "progress_imatrix": "computing the importance matrix",
    "progress_quantize": "quantizing to {type}",
    "progress_perplexity": "measuring perplexity",
    "progress_galton": "Galton run {run}: {state}",
    "progress_eval_suite": "preparing the suite of held-out records in Galton",
    "eval_suite_reused": "Reusing Galton's suite {name} ({n} cases).",
    "eval_suite_created": "Built Galton's suite {name} with {n} cases from the held-out records.",
    "eval_records_capped": "Only {n} of the {total} held-out records were used (setting galton.eval_cases).",
    "eval_ref_explicit": "Compared with {model}, as asked.",
    "eval_ref_parent": "Compared with its parent {model}: the difference measured is the file format, not a training.",
    "eval_ref_base": "Compared with the base model {model} at {quant}.",
    "eval_ref_base_build": "Compared with the base model {model} at {quant} (it will be prepared first, as it does not exist yet).",
    "eval_ref_context": "Compared with {model} without the context extension, at {quant}.",
    "eval_ref_context_build": "Compared with {model} without the context extension, at {quant} (it will be prepared first, as it does not exist yet).",
    "eval_ref_imatrix": "The reference uses an importance matrix computed on the same calibration text as the result.",
    "eval_ref_unknown_base": "The model this result was trained from is not in the lineage, so it is compared with its parent file, which measures the quantization more than the training.",
    "eval_pending_judge": "{n} answers still wait for the judge; the verdict rests on the others.",
    "note_rope_missing": "The converter wrote no rope scaling into the GGUF; run it with --rope-scaling yarn --rope-scale N --yarn-orig-ctx M (see publish_llama extra_args).",
    # -- misc
    "quant_needs_imatrix": "{type} without an importance matrix gives poor results; turn the matrix on.",
    "convert_note_no_script": "convert_hf_to_gguf.py was not found; set llama.src_dir in Settings.",
    "convert_note_no_arches": "The conversion script lists no architectures (unknown version).",
    "convert_note_no_arch": "The model's config.json names no architecture.",
    "convert_note_unregistered": "{arch} is not registered in this version of the conversion script; update llama.cpp.",
    # -- datasets
    "note_bad_lines": "{label}: {bad} lines were not valid JSON objects and were skipped.",
    "note_folder_many": "The folder has more than {max} files; the rest were not read.",
    "note_file_big": "{name} is larger than {mb} MB and was skipped.",
    "note_items_unmapped": "{label}: {skipped} items lacked a mapped field and were skipped.",
    "note_teacher_failed": "The teacher failed on chunk {chunk}: {detail}",
    "synthetic_note": "Estimate: about 25 s per call with a 27B model; the first call shows the real speed.",
    "progress_source": "source {n}/{total}: {type}",
    "progress_synthetic": "synthetic {n}/{total}",
    "path_invalid": "not a valid path",
    "path_relative": "the path must be absolute",
    "path_unresolved": "the path cannot be resolved",
    "folder_missing": "the folder does not exist",
    "path_root": "a drive root is too broad",
    "path_home": "the user profile folder is too broad: pick a subfolder",
    "path_system": "system folders are not allowed",
    "path_own_data_folder": "the app's own data folder cannot be read",
    "path_own_data_file": "the app's own data folder is off limits",
    "path_hidden": "configuration and hidden folders are not allowed",
    "file_missing": "the file does not exist",
    "path_credentials": "that looks like a credentials file",
    "detail_offline": "offline",
    "env_trainer": "trainer environment",
    "env_not_found": "not found",
    "env_not_in_src": "not found in llama.src_dir",
    "env_ollama_missing": "not found (only needed to publish to Ollama)",
    "env_unavailable": "unavailable",
    "env_lib_missing": "missing",
    "env_lib_old": "version {version} is older than {minimum}",
    "env_no_cuda": "no CUDA device is visible (CPU-only build, or no driver)",
    "env_fix_llama_bin": "Download a llama.cpp release for your GPU and set llama.bin_dir to its folder.",
    "env_fix_ollama": "Install Ollama from its website, or set publish.ollama_exe.",
    # -- workers
    "tip_oom": "The GPU ran out of memory. Lower the batch size or the sequence length, use QLoRA, or free the GPU.",
    "tip_module": "Install it in the trainer environment: pip install {module}.",
    "tip_gated": "This repository is gated: accept its licence on Hugging Face and save your token in Settings.",
    "tip_disk": "The disk is full: free space in the work folder.",
    "tip_cuda": "The trainer environment has no working CUDA build of PyTorch. Run env_check for the install command.",
    "tip_arch": "This transformers version cannot load that architecture. Update transformers in the trainer environment.",
    # -- extra
    "merge_problem_lacks": "model {idx} lacks {n} tensors of the first (e.g. {name})",
    "merge_problem_extra": "model {idx} has {n} tensors the first lacks (e.g. {name})",
    "merge_problem_shape": "{name}: shape {first} in the first, {other} in model {idx}",
}


def fields_of(template: str) -> set[str]:
    """The placeholder names of a template (``{{`` is a literal brace)."""
    return set(re.findall(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}", template.replace("{{", "\0\0")))


def fill(template: str, params: dict[str, Any]) -> str:
    """``template`` with ``params`` filled in; a missing parameter leaves its placeholder, so a message is never lost to a typo."""
    class _Safe(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_Safe(params))


def scalar(value: Any) -> Any:
    """A parameter value the client can print: numbers and text stay, lists are joined, everything else becomes text."""
    if isinstance(value, bool) or value is None:
        return "" if value is None else str(value).lower()
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(scalar(v)) for v in value)
    return str(value)


class CodedText(str):
    """An English sentence that remembers its ``key`` and ``params``: it behaves as a ``str`` everywhere, and ``item()`` gives the wire form."""

    key: str
    params: dict[str, Any]

    def __new__(cls, text: str, key: str = "", params: Optional[dict[str, Any]] = None) -> "CodedText":
        obj = super().__new__(cls, text)
        obj.key = key
        obj.params = dict(params or {})
        return obj

    def item(self) -> dict[str, Any]:
        return {"key": self.key, "params": {k: wire(v) for k, v in self.params.items()}, "text": str(self)}


def text(key: str, **params: Any) -> CodedText:
    """The coded sentence ``TEXTS[key]`` filled in with ``params``."""
    template = TEXTS[key]
    clean = {k: scalar(v) for k, v in params.items()}
    return CodedText(fill(template, clean), key, clean)


def error_text(key: str, **params: Any) -> CodedText:
    """The message of the catalogue error ``key`` as a ``CodedText`` (for a sentence that is stored, not raised)."""
    message, _hint = ERRORS[key]
    clean = {k: scalar(v) for k, v in params.items()}
    return CodedText(fill(message, clean), key, clean)


def hint_text(key: str, **params: Any) -> CodedText:
    """The hint of the catalogue error ``key`` as a ``CodedText`` whose key is ``hint_<key>`` (empty when the error has no hint)."""
    _message, hint = ERRORS[key]
    clean = {k: scalar(v) for k, v in params.items()}
    return CodedText(fill(hint, clean), f"hint_{key}" if hint else "", clean)


def item(value: Any) -> dict[str, Any]:
    """The wire form of a message that may be a ``CodedText``, a plain string (a legacy row, free text) or already an item."""
    if isinstance(value, CodedText) and value.key:
        return value.item()
    if isinstance(value, dict) and "text" in value:
        return {"key": value.get("key", ""), "params": value.get("params", {}), "text": str(value["text"])}
    return {"key": "", "params": {}, "text": str(value)}


def items(values: Iterable[Any]) -> list[dict[str, Any]]:
    return [item(v) for v in values]


def error_item(code: str, key: str, **params: Any) -> dict[str, Any]:
    """The wire form of an error from the catalogue: ``{key, params, text, hint}``."""
    message, hint = ERRORS[key]
    clean = {k: scalar(v) for k, v in params.items()}
    return {"code": code, "key": key, "params": clean, "text": fill(message, clean), "hint": fill(hint, clean)}


def hint_of(value: Any) -> str:
    """The hint that goes with a coded error message (what to do about it), or an empty string."""
    key = getattr(value, "key", "")
    if key in ERRORS:
        return fill(ERRORS[key][1], getattr(value, "params", {}))
    return ""


# ------------------------------------------------------------------------------------------------- recognising stored sentences
# A run keeps its sentences as plain English text (what an assistant reads). To show them in the language of the UI they are matched back to
# the catalogue: every template becomes an anchored pattern whose placeholders take the parameter values. Rows written before the keys
# existed are recognised the same way. A template that is only a placeholder ("{reason}") would match anything and is never recognised.
def _pattern(template: str) -> tuple["re.Pattern[str]", tuple[str, ...], int]:
    pieces: list[str] = []
    names: list[str] = []
    literal = 0
    for part in re.split(r"(\{\{|\}\}|\{[a-z_][a-z0-9_]*\})", template):
        if part in ("{{", "}}"):
            pieces.append(re.escape(part[0]))
            literal += 1
        elif part.startswith("{") and part.endswith("}"):
            names.append(part[1:-1])
            pieces.append(f"(?P<p{len(names) - 1}>.+?)")
        elif part:
            pieces.append(re.escape(part))
            literal += len(part)
    return re.compile("^" + "".join(pieces) + "$", re.DOTALL), tuple(names), literal


Pattern = tuple["re.Pattern[str]", str, tuple[str, ...]]
_PATTERNS: dict[str, list[Pattern]] = {}


def _compiled(kind: str) -> list[Pattern]:
    """The patterns of ``kind``: ``"text"`` (sentences: texts and error messages) or ``"hint"`` (the hints of errors), longest literal first."""
    if kind not in _PATTERNS:
        found = []
        if kind == "text":
            sources = [(key, template) for key, template in TEXTS.items()] + [(key, pair[0]) for key, pair in ERRORS.items()]
        else:
            sources = [(f"hint_{key}", pair[1]) for key, pair in ERRORS.items() if pair[1]]
        for key, template in sources:
            rx, names, literal = _pattern(template)
            if literal:
                found.append((literal, rx, key, names))
        found.sort(key=lambda f: -f[0])
        _PATTERNS[kind] = [(rx, key, names) for _literal, rx, key, names in found]
    return _PATTERNS[kind]


def recognise(value: Any, _depth: int = 0) -> Any:
    """``value`` as a ``CodedText`` when it is a sentence (or a hint) of the catalogue (a stored run keeps plain text), otherwise unchanged."""
    if not isinstance(value, str) or isinstance(value, CodedText) or not 4 <= len(value) <= 2000:
        return value
    for kind in ("text", "hint"):
        for rx, key, names in _compiled(kind):
            found = rx.match(value)
            if found:
                params = {name: _number(found.group(f"p{i}")) for i, name in enumerate(names)}
                if _depth < 3:        # a parameter may be a sentence itself ("Next step failed: <reason>")
                    params = {k: recognise(v, _depth + 1) if isinstance(v, str) else v for k, v in params.items()}
                return CodedText(value, key, params)
    return value


def recognise_all(values: Iterable[Any]) -> list[Any]:
    return [recognise(v) for v in values]


# The fields of the API's answers that carry a message, and the ones that hold the user's own data (never touched).
MESSAGE_FIELDS = frozenset({"title", "error", "hint", "message", "note", "notes", "warnings", "detail", "gguf_error", "convert_note", "reason",
                            "problem", "fix", "problems", "why", "formula"})
USER_DATA = frozenset({"record", "records", "preview", "params", "recipe", "spec", "log_tail", "sources", "source", "samples", "rows", "modelfile"})


def localise(value: Any, field: str = "") -> Any:
    """``value`` with the stored English sentences in its message fields recognised, so that ``wire`` can send them as ``{key, params, text}``.
    Only used for the bundled UI; assistants keep the plain sentences. Records and parameters (the user's data) are left alone."""
    if isinstance(value, dict):
        return {k: (v if k in USER_DATA else localise(v, k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [localise(v, field) for v in value]
    if field in MESSAGE_FIELDS and isinstance(value, str):
        return recognise(value)
    return value


def _number(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def wire(value: Any) -> Any:
    """``value`` as the bundled UI receives it: every ``CodedText`` becomes ``{key, params, text}`` so the client can translate it. The REST agent
    route and MCP never go through this: assistants get the plain English sentences."""
    if isinstance(value, CodedText):
        return value.item() if value.key else str(value)
    if isinstance(value, dict):
        return {k: wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(v) for v in value]
    return value
