# Pygmalion's Hoard - agent tools

Every tool is served by the app at `GET /api/agent/tools` and `POST /api/agent/call` (Bearer token from `data/mcp-token`), by the stdio bridge `mcp_server.py`, and to the bundled UI through `POST /api/ui/call`. The argument tables are generated from the code (`python scripts/gen_api_doc.py`).

## `pygmalion_overview`

Studio at a glance: environment, GPUs, running jobs, latest artifacts. Resumen de Pygmalion.

Start here. Lists what is missing and the next sensible step.
Sinónimos: qué puedo hacer, estado del estudio, trabajos en marcha, modelos, entrenamientos

Annotations: readOnlyHint, idempotentHint.

## `env_check`

Check the trainer environment, llama.cpp tools and Ollama and how to fix gaps. Comprobar entorno.

Runs the probe in the trainer's Python: versions, CUDA, bitsandbytes, free disk; finds llama-quantize, llama-imatrix and the convert scripts.
Sinónimos: instalar, falta torch, CUDA, bitsandbytes, llama.cpp, configuración, diagnóstico

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `fresh` (boolean) | no | Run the probe again (a few seconds) instead of answering from the last check. |

## `bases_list`

Base models on disk with size, architecture, trainable and convertible badges. Modelos base locales.

Sinónimos: qué modelos tengo, carpetas hf, bases, descargados

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `deep` (boolean) | no | Also ask the trainer environment which architectures transformers can load (slower). |

## `base_get`

One base model: config, memory estimate, support check, derived artifacts. Detalle de un modelo base.

Sinónimos: arquitectura, contexto, parámetros, ¿se puede entrenar?, ¿se puede convertir?

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `base` (string) | yes | Base model artifact id, name or folder path. |

## `hf_search`

Search Hugging Face for base models by text, with size, licence and gating. Buscar modelos en Hugging Face.

Network use. Sorted by downloads by default.
Sinónimos: buscar modelo, descargar qwen, llama, gemma, más descargados

Annotations: readOnlyHint, idempotentHint, openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `query` (string) | yes | Text to search, e.g. 'qwen 4b instruct'. |
| `pipeline` (text-generation \| image-text-to-text) | no |  |
| `sort` (downloads \| likes \| lastModified \| trendingScore) | no |  |
| `limit` (integer) | no |  |

## `base_download`

Download a base model from Hugging Face into the work folder (confirm=true after the size). Descargar modelo.

Without confirm it only reports the download size and the free disk. Resumable. Only safetensors, configs and tokenizer files.
Sinónimos: bajar modelo, traer de Hugging Face, descarga

Annotations: idempotentHint, openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `repo_id` (string) | yes | Hugging Face repository, e.g. 'Qwen/Qwen3-4B'. |
| `revision` (string) | no |  |
| `confirm` (boolean) | no | Without it the call only reports the download size. Repeat with true after showing it. |

## `datasets_list`

List datasets with their versions, record counts and tokens. Lista de datasets.

Sinónimos: mis datos, conjuntos de entrenamiento, corpus

Annotations: readOnlyHint, idempotentHint.

## `dataset_get`

One dataset with its versions, stats, recipe and split. Detalle de un dataset.

Sinónimos: histograma de longitudes, balance de roles, versiones, receta, PII

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `dataset` (string) | yes | Dataset id or name (or a version id). |
| `n` (integer/null) | no | Version number; the latest when omitted. |

## `dataset_create`

Build a dataset version from files, folders, JSONL/CSV, family apps or a teacher model. Crear dataset.

Sources combine in one build; operations (dedupe, length, language, PII) run in order; the split is seeded. Immutable versions.
Sinónimos: preparar datos, juntar textos, destilar, sintético, importar JSONL, usar mis escritos

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `name` (string) | no | Name of a new dataset (letters, digits, spaces, dots and hyphens). |
| `dataset` (string) | no | Add a new version to this existing dataset instead of creating one. |
| `description` (string) | no |  |
| `kind` (string/null) | no | Force the record kind; detected from the sources when omitted. |
| `sources` (array) | yes | Sources to combine. Each has a `type`: jsonl {text\|path}, csv {text\|path, columns:{prompt,response,text}}, files {paths}, folder {path, extensions}, family {app, tool, arguments, items_path, mapping:{prompt,response,text}}, synthetic {task, from:[sources], max_items, per_chunk}. |
| `operations` (array) | no | Operations in order: {op:'dedupe_exact'}, {op:'dedupe_near', threshold}, {op:'length', min_chars, max_chars}, {op:'language', keep:['es','en']}, {op:'pii', mode:'mask'\|'drop'}. |
| `split` (object) | no | {eval_pct, min_eval, seed}; defaults 5 %, 20 records, seed 42. |
| `wait_s` (number) | no | Seconds to wait for the build; it keeps running in the background after that. |

## `dataset_records`

Page through a dataset version's records with filters. Registros de un dataset.

Sinónimos: ver ejemplos, filtrar por estado, buscar en el dataset, sintéticos pendientes

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `dataset` (string) | yes |  |
| `n` (integer/null) | no |  |
| `offset` (integer) | no |  |
| `limit` (integer) | no |  |
| `status` ( \| ok \| pending \| rejected) | no |  |
| `text` (string) | no |  |
| `source` (string) | no |  |
| `synthetic` (boolean/null) | no |  |
| `split` ( \| train \| eval \| none) | no |  |
| `lang` ( \| es \| en) | no |  |
| `min_chars` (integer) | no |  |
| `max_chars` (integer) | no |  |
| `has_pii` (boolean/null) | no |  |

## `dataset_review`

Accept, reject or edit records; the result is a new version. Revisar dataset.

Sinónimos: aceptar, rechazar, corregir ejemplos, aprobar sintéticos

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `dataset` (string) | yes |  |
| `n` (integer/null) | no |  |
| `accept` (array) | no | Record indexes to accept. |
| `reject` (array) | no | Record indexes to reject. |
| `edit` (object) | no | Index to the corrected record, in the dataset's own format. |
| `accept_all_pending` (boolean) | no | Accept every pending (synthetic) record. |

## `dataset_preview_source`

Show what a source would yield (first items, kind, notes) without building. Vista previa de una fuente.

For a family app it shows what the tool returned; for a synthetic source it estimates time and cost.
Sinónimos: probar fuente, qué sale de esta carpeta, qué devuelve la app

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `source` (object) | yes | One source as in dataset_create. A synthetic source only estimates unless sample=true. |
| `limit` (integer) | no |  |

## `dataset_apply`

Apply operations (dedupe, length, language, PII) to a version; makes a new version. Limpiar dataset.

Sinónimos: deduplicar, filtrar por longitud, enmascarar datos personales, idioma

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `dataset` (string) | yes |  |
| `n` (integer/null) | no |  |
| `operations` (array) | yes |  |

## `dataset_delete`

Delete a dataset and its versions (confirm=true). Borrar dataset.

Artifacts trained on it keep their lineage.
Sinónimos: eliminar datos

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `dataset` (string) | yes |  |
| `confirm` (boolean) | no |  |

## `train_plan`

Plan a LoRA/QLoRA run: parameters, memory and time estimate, GPU that fits. Plan de entrenamiento.

The estimate shows its formula. Call before train_start.
Sinónimos: cuánta VRAM, cuánto tarda, qué rango, qué learning rate, cabe en la GPU

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `method` (string/null) | no |  |
| `rank` (integer/null) | no |  |
| `alpha` (integer/null) | no |  |
| `dropout` (number/null) | no |  |
| `lr` (number/null) | no |  |
| `seq_len` (integer/null) | no |  |
| `batch` (integer/null) | no |  |
| `grad_accum` (integer/null) | no |  |
| `epochs` (number/null) | no |  |
| `max_steps` (integer/null) | no |  |
| `warmup` (number/null) | no |  |
| `weight_decay` (number/null) | no |  |
| `eval_every` (integer/null) | no |  |
| `save_every` (integer/null) | no |  |
| `seed` (integer/null) | no |  |
| `target_modules` (array/null) | no | Last names of the layers to adapt (q_proj, v_proj...) or ['all']. |
| `train_on` (string/null) | no |  |
| `gradient_checkpointing` (boolean/null) | no |  |
| `base` (string) | yes | Base model artifact id, name or folder path. |
| `dataset` (string) | yes | Dataset id or name (or a version id). |
| `dataset_n` (integer/null) | no | Dataset version number; the latest when omitted. |

## `train_start`

Start a LoRA/QLoRA fine-tune (optionally the full recipe after it). Entrenar un modelo.

Queues a pipeline: train, then the steps in `after` (merge, convert, quantize, publish, evaluate). Waits for an allowed GPU lease.
Sinónimos: ajustar, afinar, fine-tune, LoRA, QLoRA, enseñar mi estilo, receta completa

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `method` (string/null) | no |  |
| `rank` (integer/null) | no |  |
| `alpha` (integer/null) | no |  |
| `dropout` (number/null) | no |  |
| `lr` (number/null) | no |  |
| `seq_len` (integer/null) | no |  |
| `batch` (integer/null) | no |  |
| `grad_accum` (integer/null) | no |  |
| `epochs` (number/null) | no |  |
| `max_steps` (integer/null) | no |  |
| `warmup` (number/null) | no |  |
| `weight_decay` (number/null) | no |  |
| `eval_every` (integer/null) | no |  |
| `save_every` (integer/null) | no |  |
| `seed` (integer/null) | no |  |
| `target_modules` (array/null) | no | Last names of the layers to adapt (q_proj, v_proj...) or ['all']. |
| `train_on` (string/null) | no |  |
| `gradient_checkpointing` (boolean/null) | no |  |
| `base` (string) | yes | Base model artifact id, name or folder path. |
| `dataset` (string) | yes | Dataset id or name (or a version id). |
| `dataset_n` (integer/null) | no | Dataset version number; the latest when omitted. |
| `name` (string) | no | Name for the adapter. |
| `after` (object/null) | no | Steps after the training: {merge: true, convert: 'f16'\|'bf16'\|'q8_0', quantize: ['Q4_K_M'], imatrix: true, calibration: {source:'bundled'\|'dataset', dataset}, perplexity: true, publish: {target:'ollama'\|'llama'\|'both', name, tag, num_ctx}, evaluate: {intent:'dataset'\|'style'\|'code'\|'context'\|'general', suites:[...], against: artifact, regression}}. |
| `force` (boolean) | no | Queue it even when the estimate does not fit the allowed GPUs. |
| `wait_s` (number) | no |  |

## `merge_check`

Check that models can be merged (same tensors, shapes, dtypes) before merging. Comprobar fusión.

Sinónimos: compatibles, misma arquitectura, formas distintas

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `models` (array) | yes | Model artifact ids, names or folder paths. |
| `base` (string) | no |  |

## `merge_lora_start`

Fold a LoRA adapter into its base model (optionally the steps after it). Fusionar LoRA con la base.

Sinónimos: mezclar adapter, merge_and_unload, aplicar LoRA

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `base` (string) | yes |  |
| `adapter` (string) | yes | Adapter artifact id or name. |
| `name` (string) | no |  |
| `device` ( \| cpu \| cuda) | no | Where to merge; the setting merge.device when empty (cpu needs no VRAM). |
| `after` (object/null) | no | Steps after the training: {merge: true, convert: 'f16'\|'bf16'\|'q8_0', quantize: ['Q4_K_M'], imatrix: true, calibration: {source:'bundled'\|'dataset', dataset}, perplexity: true, publish: {target:'ollama'\|'llama'\|'both', name, tag, num_ctx}, evaluate: {intent:'dataset'\|'style'\|'code'\|'context'\|'general', suites:[...], against: artifact, regression}}. |
| `wait_s` (number) | no |  |

## `merge_models_start`

Merge models of the same architecture: linear, slerp, ties or dare. Fusionar modelos.

Streams tensor by tensor over safetensors shards; never loads a whole model.
Sinónimos: mezclar modelos, model soup, slerp, TIES, DARE, promedio de pesos

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `models` (array) | yes |  |
| `method` (linear \| slerp \| ties \| dare) | no |  |
| `weights` (array/null) | no | One weight per model (linear, ties, dare). |
| `base` (string) | no | Required by ties and dare: the model the others were tuned from. |
| `t` (number) | no | slerp: interpolation between the first and the second model. |
| `t_map` (object) | no | slerp: {regex on tensor name: t} overrides. |
| `density` (number) | no | ties and dare: fraction of each task vector kept. |
| `lam` (number) | no | ties: scale of the merged task vector. |
| `seed` (integer) | no |  |
| `consensus` (linear \| ties) | no |  |
| `normalize` (boolean) | no |  |
| `name` (string) | no |  |
| `after` (object/null) | no | Steps after the training: {merge: true, convert: 'f16'\|'bf16'\|'q8_0', quantize: ['Q4_K_M'], imatrix: true, calibration: {source:'bundled'\|'dataset', dataset}, perplexity: true, publish: {target:'ollama'\|'llama'\|'both', name, tag, num_ctx}, evaluate: {intent:'dataset'\|'style'\|'code'\|'context'\|'general', suites:[...], against: artifact, regression}}. |
| `wait_s` (number) | no |  |

## `ctx_extend_start`

Make a copy of a model with YaRN rope scaling for a longer context. Ampliar el contexto.

No weights change; quality at the far end must be measured (evaluate with intent=context).
Sinónimos: contexto largo, YaRN, 32k, 128k, ventana de contexto

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `model` (string) | yes |  |
| `factor` (number/null) | no | YaRN factor: 2, 4 or 8. |
| `target_length` (integer/null) | no | Alternative to factor: the context length wanted. |
| `name` (string) | no |  |
| `after` (object/null) | no | Steps after the training: {merge: true, convert: 'f16'\|'bf16'\|'q8_0', quantize: ['Q4_K_M'], imatrix: true, calibration: {source:'bundled'\|'dataset', dataset}, perplexity: true, publish: {target:'ollama'\|'llama'\|'both', name, tag, num_ctx}, evaluate: {intent:'dataset'\|'style'\|'code'\|'context'\|'general', suites:[...], against: artifact, regression}}. |
| `wait_s` (number) | no |  |

## `ctx_fit`

KV-cache memory per context length for a GGUF and whether it fits the allowed GPUs. Tabla de contexto.

Sinónimos: cuánto contexto cabe, memoria KV, caché, longitud máxima

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `gguf` (string) | yes |  |
| `contexts` (array/null) | no | Context lengths to tabulate. |
| `gpu` (integer/null) | no | Only this allowed GPU. |

## `convert_start`

Convert a Hugging Face model (or a LoRA adapter) to GGUF. Convertir a GGUF.

Checks that the installed llama.cpp knows the architecture.
Sinónimos: pasar a GGUF, f16, bf16, convert_hf_to_gguf, adapter GGUF

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `model` (string) | no | Hugging Face model artifact to convert. |
| `adapter` (string) | no | Convert a LoRA adapter to a GGUF adapter instead (needs its base). |
| `base` (string) | no |  |
| `outtype` (f16 \| bf16 \| q8_0) | no |  |
| `name` (string) | no |  |
| `after` (object/null) | no | Steps after the training: {merge: true, convert: 'f16'\|'bf16'\|'q8_0', quantize: ['Q4_K_M'], imatrix: true, calibration: {source:'bundled'\|'dataset', dataset}, perplexity: true, publish: {target:'ollama'\|'llama'\|'both', name, tag, num_ctx}, evaluate: {intent:'dataset'\|'style'\|'code'\|'context'\|'general', suites:[...], against: artifact, regression}}. |
| `wait_s` (number) | no |  |

## `quantize_start`

Quantize a GGUF (several types at once) with an importance matrix from your own text. Cuantizar.

Types: Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, Q3_K_M, IQ3_M. The matrix runs on a leased allowed GPU; IQ types are poor without it.
Sinónimos: comprimir, Q4_K_M, imatrix, matriz de importancia, calibración, reducir tamaño

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `gguf` (string) | no | An f16, bf16 or q8_0 GGUF artifact. |
| `model` (string) | no | Alternatively a Hugging Face model: it is converted first. |
| `types` (array) | no | Any of Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, Q3_K_M, IQ3_M. |
| `imatrix` (boolean) | no | Compute an importance matrix first (needs a GPU lease); IQ types are poor without it. |
| `calibration` (object/null) | no | {source:'bundled'} or {source:'dataset', dataset, n}: the text the matrix is computed on. |
| `chunks` (integer/null) | no |  |
| `perplexity` (boolean) | no | Measure perplexity of the first type afterwards. |
| `outtype` (f16 \| bf16 \| q8_0) | no |  |
| `name` (string) | no |  |
| `wait_s` (number) | no |  |

## `perplexity_start`

Measure the perplexity (PPL ± error) of a GGUF on a text. Medir perplejidad.

Sinónimos: PPL, calidad tras cuantizar, comparar cuantizaciones

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `gguf` (string) | yes |  |
| `text` (object/null) | no | {source:'bundled'} or {source:'dataset', dataset, n}. |
| `ctx` (integer/null) | no |  |
| `chunks` (integer/null) | no |  |
| `wait_s` (number) | no |  |

## `jobs_list`

Queue and history of jobs with state, progress and ETA. Lista de trabajos.

Sinónimos: cola, qué está corriendo, historial, fallidos, interrumpidos

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `states` (array/null) | no | Any of queued, waiting_gpu, running, done, failed, cancelled, interrupted. |
| `kind` (string) | no | One of download, dataset_build, train, merge_lora, merge_models, convert, imatrix, quantize, perplexity, ctx_extend, publish, evaluate. |
| `limit` (integer) | no |  |

## `job_get`

One job: progress, loss curve, VRAM estimate, log tail, pipeline. Detalle de un trabajo.

Sinónimos: cómo va el entrenamiento, loss, curva, log, ETA, error

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `job` (string) | yes |  |
| `log_lines` (integer) | no |  |
| `curve_points` (integer) | no |  |

## `job_cancel`

Cancel a job (a training run saves a checkpoint first). Cancelar trabajo.

Sinónimos: parar, detener entrenamiento

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `job` (string) | yes |  |

## `job_resume`

Resume an interrupted, failed or cancelled job (training continues from its last checkpoint). Reanudar.

Sinónimos: continuar, reintentar, retomar tras reinicio

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `job` (string) | yes |  |

## `job_delete`

Delete a finished job and its log (confirm=true); artifacts are kept. Borrar trabajo.

Sinónimos: limpiar historial

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `job` (string) | yes |  |
| `confirm` (boolean) | no |  |

## `artifacts_list`

List artifacts: bases, adapters, merges, GGUF files, matrices, Ollama tags. Lista de artefactos.

Sinónimos: mis modelos, resultados, ficheros GGUF, adaptadores, publicados

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `kind` (string) | no | One of base, adapter, merged, gguf, imatrix, ollama, ctx_variant. |
| `text` (string) | no |  |
| `pinned` (boolean/null) | no |  |
| `limit` (integer) | no |  |

## `artifact_get`

One artifact with its lineage, recipe, metrics and Galton verdict. Detalle de un artefacto.

Sinónimos: de dónde viene, con qué datos, parámetros, evaluación, linaje

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `artifact` (string) | yes | Artifact id or name. |

## `artifact_recipe`

Reproducible recipe of an artifact (every step from the base, dataset hashes, parameters). Receta.

Sinónimos: exportar receta, reproducir, JSON de la receta

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `artifact` (string) | yes | Artifact id or name. |

## `artifact_update`

Rename an artifact, edit its notes or pin it. Editar artefacto.

Sinónimos: anclar, notas, renombrar, favorito

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `artifact` (string) | yes |  |
| `name` (string/null) | no |  |
| `notes` (string/null) | no |  |
| `pinned` (boolean/null) | no |  |

## `artifact_delete`

Delete an artifact record (and optionally its files) (confirm=true). Borrar artefacto.

Published artifacts must be unpublished first; downloaded bases keep their files.
Sinónimos: eliminar modelo, liberar disco

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `artifact` (string) | yes |  |
| `delete_files` (boolean) | no | Also delete the files (never for a base model). |
| `confirm` (boolean) | no |  |

## `lineage_graph`

Nodes and edges of the artifact graph in columns by kind. Grafo de linaje.

Sinónimos: árbol, familia de modelos, genealogía

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `root` (string) | no | Limit to the family of this artifact. |
| `kinds` (array/null) | no |  |

## `evaluate_plan`

Say what an evaluation would compare with, and whether that file must be prepared first. Plan de evaluación.

Read-only. The reference is the base model the training started from, in the same quantization as the result, for the intents that ask whether a training helped.
Sinónimos: con qué se compara, modelo base, referencia, cuantización, antes de evaluar

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `artifact` (string) | yes | A GGUF or published Ollama artifact. |
| `against` (string) | no | A file to compare with; it always wins over the rule of the intent. |
| `intent` ( \| dataset \| style \| writing \| code \| context \| general \| smoke \| quant) | no | What to check. dataset: the held-out records of the dataset the result was trained on, graded by Galton's judge (the default for such a result). dataset, style, writing, code, general and smoke ask whether a training helped: the reference is the base model the training started from, in the same quantization (prepared first when it does not exist). context: the model before the context extension. quant: a plain quantization, compared with the file it was made from. |
| `suites` (array/null) | no | Galton suites; chosen from the intent when omitted. |
| `regression` (boolean) | no | With intent=dataset, also run Galton's quick general suite. |

## `evaluate_start`

Measure a GGUF or Ollama result against its reference with Galton's Hoard. Evaluar con Galton.

Verdict better, worse or no clear difference with deltas and intervals; stored on the artifact. The reference is the base model the training started from in the same quantization (queued first as a pipeline when it does not exist), the file before a context extension, or the parent file of a plain quantization; see evaluate_plan. A result trained on a dataset is measured on that dataset's held-out records (intent dataset, graded by Galton's judge) plus a quick general suite for regressions; other intents pick suites.
Sinónimos: ¿es mejor que el original?, comparar, regresión, medir, prueba de estilo, contexto largo, registros reservados del dataset

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `artifact` (string) | yes | A GGUF or published Ollama artifact. |
| `against` (string) | no | A file to compare with (always wins). Empty: the reference of the intent, see evaluate_plan. |
| `intent` ( \| dataset \| style \| writing \| code \| context \| general \| smoke \| quant) | no | What to check. dataset: the held-out records of the dataset the result was trained on, graded by Galton's judge (the default for such a result). dataset, style, writing, code, general and smoke ask whether a training helped: the reference is the base model the training started from, in the same quantization (prepared first when it does not exist). context: the model before the context extension. quant: a plain quantization, compared with the file it was made from. |
| `suites` (array/null) | no | Galton suites; chosen from the intent when omitted. |
| `regression` (boolean) | no | With intent=dataset, also run Galton's quick general suite. |
| `wait_s` (number) | no |  |

## `publish_ollama`

Publish a GGUF as an Ollama model tagged pyg-<name>:<tag>. Publicar en Ollama.

Writes a Modelfile and runs ollama create.
Sinónimos: ollama create, usar en Faustus, exponer el modelo

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `gguf` (string) | yes |  |
| `name` (string) | no | Name part of the tag; the tag is always pyg-<name>:<tag>. |
| `tag` (string) | no |  |
| `num_ctx` (integer/null) | no |  |
| `adapter` (string) | no | A GGUF adapter artifact to load on top (ADAPTER). |
| `template` (string) | no | Chat template for the Modelfile when the GGUF has none. |
| `system` (string) | no |  |
| `wait_s` (number) | no |  |

## `publish_llama`

Add a GGUF as a llama.cpp server the hub can start (never started automatically). Publicar en llama.cpp.

Sinónimos: llama-server, backend, puerto, hub, arrancar servidor

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `gguf` (string) | yes |  |
| `name` (string) | no |  |
| `ctx` (integer/null) | no |  |
| `ngl` (integer) | no |  |
| `gpu` (integer/null) | no | Allowed GPU for the server; the first allowed one when omitted. |
| `extra_args` (array) | no | Extra llama-server flags, e.g. ['--rope-scaling','yarn']. |
| `wait_s` (number) | no |  |

## `unpublish`

Remove a published Ollama tag or llama.cpp backend (confirm=true). Retirar publicación.

Sinónimos: ollama rm, quitar del hub, despublicar

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `artifact` (string) | yes | The published GGUF or the Ollama artifact. |
| `target` (all \| ollama \| llama) | no |  |
| `name` (string) | no |  |
| `confirm` (boolean) | no |  |

## `settings_get`

Settings: allowed GPUs, paths, defaults for training, Galton link, publishing. Ajustes.

Sinónimos: configuración, GPUs permitidas, rutas, valores por defecto

Annotations: readOnlyHint, idempotentHint.

## `settings_set`

Change settings (allowed GPUs need confirm_reserved for the owner's GPUs). Cambiar ajustes.

Sinónimos: rango LoRA, learning rate, carpeta de trabajo, llama.cpp, Python del entorno

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `values` (object) | yes | Setting key to value, e.g. {'train.rank': 32}. |
| `confirm_reserved` (boolean) | no | Allow a GPU that belongs to the owner. Only when the owner said so. |

## `secret_set`

Store the Hugging Face token (write-only, never shown back). Guardar token.

Sinónimos: token de Hugging Face, modelos con acceso restringido

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `name` (string) | no |  |
| `value` (string) | no | The Hugging Face token; empty removes it. It is never shown back. |

## `gpu_status`

Per GPU: memory used and free, allowed or reserved, who holds leases. Estado de las GPU.

Sinónimos: VRAM libre, quién usa la GPU, leases, cola del hub

Annotations: readOnlyHint, idempotentHint.

## REST routes for the UI

- `GET /api/health`, `GET /api/status`
- `GET /api/dashboard` - GPUs, running and queued jobs, recent artifacts, environment state, Galton and hub availability.
- `GET /api/ui/tools` - the catalogue for the UI.
- `POST /api/ui/call` `{name, arguments}` - any tool above, uncapped (the UI is local).
- `GET /manifest.webmanifest`, `GET /sw.js` - installable app.
