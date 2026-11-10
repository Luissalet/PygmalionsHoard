# Galton's Hoard — agent tools

Every tool is served by the app at `GET /api/agent/tools` and `POST /api/agent/call` (Bearer token from `data/mcp-token`), by the stdio bridge `mcp_server.py`, and to the bundled UI through `POST /api/ui/call`. The argument tables are generated from the code (`python scripts/gen_api_doc.py`).

## `galton_overview`

Models, routing table, running run, notices, GPUs and next steps. Estado general del banco de pruebas.

Start here: what was measured, what is new or stale, what the routing table says, whether a run is in progress.
Sinónimos: resumen, qué modelos tengo, qué modelo uso, mediciones, pruebas, novedades, regresiones

Annotations: readOnlyHint, idempotentHint.

## `galton_status`

Health: scheduler, watch, llama-server, GPUs, settings. Estado de Galton.

Sinónimos: configuración, planificador, vigilancia, ajustes, servidor llama.cpp

Annotations: readOnlyHint, idempotentHint.

## `models_list`

List models (servers, Ollama, GGUF files) with measured/stale flags. Lista de modelos.

Filter by text, kind, vision, enabled, measured (never/stale/fresh).
Sinónimos: qué modelos hay, instalados, cuáles no he medido, cuantizaciones, visión

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `text` (string) | no | Matches name, family, quantisation, provider or alias. |
| `kind` ( \| server \| gguf) | no |  |
| `enabled` (boolean/null) | no |  |
| `vision` (boolean/null) | no |  |
| `measured` (any \| never \| stale \| fresh) | no | never: no results; stale: the file changed since it was measured; fresh: measured at its current version. |
| `include_missing` (boolean) | no | Also list models that are no longer installed. |
| `include_adhoc` (boolean) | no | Also list evaluation-only entries created from specs. |
| `limit` (integer) | no |  |

## `models_refresh`

Rediscover models: Ollama, llama-server ports, GGUF folders. Buscar modelos nuevos.

Detects new models, changed files (stale results), models that disappeared and servers that now serve another model (the old entry is marked not served, never deleted).
Sinónimos: escanear, actualizar lista, detectar modelos, descubrir

Annotations: idempotentHint, openWorldHint.

## `model_get`

One model: aliases, metadata, where it would run, scores, speed, memory. Detalle de un modelo.

Sinónimos: cuánto ocupa, qué tal rinde, alias, nombres, cuántos tokens por segundo

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `model` (string) | yes | Model id (c_…), name or any alias. |

## `model_add`

Register a server URL or a GGUF file to measure. Añadir un modelo.

A GGUF is run by Galton with its own llama-server on an allowed GPU; a server is called as it is. Remote endpoints stay off until remote_ok.
Sinónimos: añadir servidor, añadir gguf, endpoint, ruta del modelo

Annotations: idempotentHint, openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `url` (string) | no | Base URL of a running server (llama-server style or Ollama), e.g. http://127.0.0.1:8081. |
| `model` (string) | no | Model name on that server (needed when the server offers several). |
| `api` ( \| chat \| ollama) | no | chat: llama-server and other chat-completions servers. Empty: detected. |
| `path` (string) | no | Absolute path of a .gguf file Galton will run itself with llama-server (instead of url). |
| `mmproj` (string) | no | Absolute path of the vision projector for that file. |
| `name` (string) | no |  |
| `remote_ok` (boolean) | no | Remote endpoints cost money and send prompts off this computer: true only if the user explicitly said so. |

## `model_update`

Enable or disable a model, rename it, add aliases, allow a remote endpoint. Editar un modelo.

Sinónimos: activar, desactivar, renombrar, alias, permitir remoto, proyector de visión

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `model` (string) | yes |  |
| `enabled` (boolean/null) | no |  |
| `name` (string/null) | no |  |
| `aliases` (array/null) | no | Replace the list of aliases. |
| `add_aliases` (array/null) | no | Names to add: every name other apps see this model under. |
| `remote_ok` (boolean/null) | no | Allow measuring a remote endpoint (cost warning: only if the user asked). |
| `vision` (boolean/null) | no |  |
| `mmproj` (string/null) | no | Absolute path of the vision projector ('' clears it). |

## `model_remove`

Remove a model and its stored results (confirm=true). Quitar un modelo.

Installed models come back on the next refresh; disabling is the gentle way.
Sinónimos: borrar modelo, eliminar

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `model` (string) | yes |  |
| `confirm` (boolean) | no |  |

## `suites_list`

List test suites with category and case count. Lista de suites de pruebas.

Built-in: escritura-es, razonamiento, matematicas, codigo-python, extraccion, herramientas, instrucciones, contexto-largo, rag-citas, vision, traduccion, resumen, rapida.
Sinónimos: baterías, conjuntos de pruebas, benchmarks, tests

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `category` (string) | no | One of writing_es, reasoning, math, code, extraction, tool_use, instruction, long_context, rag, vision, translation, summary, custom. |
| `builtin` (boolean/null) | no | true: only the built-in suites; false: only yours. |

## `suite_get`

One suite with its cases (prompt preview, checker, weight). Detalle de una suite.

Sinónimos: casos, preguntas, pruebas de una suite

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `suite` (string) | yes | Suite id (s_…) or name. |
| `include_cases` (boolean) | no |  |
| `limit` (integer) | no |  |
| `offset` (integer) | no |  |

## `case_get`

One case in full: prompt, tools, checker, reference answer, result count. Detalle de un caso.

Sinónimos: ver caso, pregunta completa, respuesta esperada

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `case` (string) | yes |  |

## `suite_create`

Create your own suite, optionally with cases. Crear una suite propia.

Categories other than custom feed the routing table once measured.
Sinónimos: nueva suite, mis pruebas, conjunto propio

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `name` (string) | yes |  |
| `description` (string) | no |  |
| `category` (string) | no | One of writing_es, reasoning, math, code, extraction, tool_use, instruction, long_context, rag, vision, translation, summary, custom. 'custom' suites never feed the routing table. |
| `max_tokens` (integer) | no | Default answer length for its cases. |
| `cases` (array) | no |  |

## `suite_update`

Rename or change the description, category or answer length of your suite. Editar suite.

Sinónimos: renombrar suite, cambiar categoría

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `suite` (string) | yes |  |
| `name` (string/null) | no |  |
| `description` (string/null) | no |  |
| `category` (string/null) | no |  |
| `max_tokens` (integer/null) | no |  |

## `suite_duplicate`

Copy any suite (also a built-in one) into an editable suite of your own. Duplicar una suite.

Sinónimos: copiar suite, partir de una suite

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `suite` (string) | yes |  |
| `name` (string) | no | Default: «<name> (copia)». |

## `suite_remove`

Delete one of your suites and its cases (confirm=true). Borrar suite propia.

Sinónimos: eliminar suite

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `suite` (string) | yes |  |
| `confirm` (boolean) | no |  |

## `case_add`

Add a case (prompt + expected answer or checker) to your suite. Añadir un caso de prueba.

Save a case from a conversation when the user asks: the prompt, what a right answer contains, and how to check it.
Sinónimos: guardar esta pregunta, nuevo caso, prueba propia, respuesta esperada, comprobador

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `title` (string) | no | Empty: taken from the prompt. |
| `prompt` (string) | no | What the model is asked. |
| `messages` (array/null) | no | A whole conversation [{role, content}] instead of prompt. |
| `system` (string) | no |  |
| `expected` (string) | no | The right answer. Without a checker: a number is compared as a number, text must be contained in the answer. |
| `checker` (object/string/null) | no | A checker object ({type: exact\|contains\|regex\|choice\|number\|math_equiv\|json\|tool_call\|python_tests\|constraints\|needle\|citations\|judge\|family\|none\|all\|any, ...}) or a shorthand: exact:Madrid, contains:uno\|dos, any:a\|b, number:42, choice:B, regex:^\d+$, math:x**2, judge:rubric, none. |
| `weight` (number) | no |  |
| `tags` (array) | no |  |
| `notes` (string) | no |  |
| `tools` (array) | no | Function definitions (chat-completions tools format) offered to the model (for tool_call cases). |
| `max_tokens` (integer/null) | no |  |
| `min_context` (integer/null) | no | Skip models whose context is smaller. |
| `image_paths` (array) | no | Absolute paths of images to attach (PNG, JPEG, WebP, GIF): the case then needs a vision model. |
| `suite` (string) | yes | One of your suites (built-in suites are read-only: duplicate them first). |

## `case_update`

Change a case of your suite: prompt, expected answer, checker, weight. Editar un caso.

If the task changed, earlier results measured the old version (forget_results deletes them).
Sinónimos: corregir caso, cambiar respuesta esperada, peso

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `case` (string) | yes | Case id (k_…). |
| `title` (string/null) | no |  |
| `prompt` (string/null) | no |  |
| `messages` (array/null) | no |  |
| `system` (string/null) | no |  |
| `expected` (string/null) | no |  |
| `checker` (object/string/null) | no |  |
| `weight` (number/null) | no |  |
| `tags` (array/null) | no |  |
| `notes` (string/null) | no |  |
| `tools` (array/null) | no |  |
| `max_tokens` (integer/null) | no |  |
| `min_context` (integer/null) | no |  |
| `image_paths` (array/null) | no |  |
| `forget_results` (boolean) | no | Delete the earlier results of this case when its prompt or checker changed (they measured the old version). |

## `case_remove`

Delete a case of your suite (confirm=true). Borrar un caso.

Sinónimos: eliminar caso, quitar pregunta

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `case` (string) | yes |  |
| `confirm` (boolean) | no |  |

## `cases_import`

Import cases from JSONL or CSV text or an absolute path. Importar casos.

Columns: prompt (required), title, system, expected, checker (full object or exact:/contains:/number:/regex:/choice:/judge: shorthand), weight, tags, notes.
Sinónimos: cargar preguntas, csv, jsonl, importar pruebas

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `suite` (string) | yes | One of your suites. |
| `text` (string) | no | JSONL (one object per line) or CSV with a header row. Columns: prompt (required), title, system, expected, checker, weight, tags, notes. |
| `path` (string) | no | Absolute path of a .jsonl, .json or .csv file instead of text. |
| `format` (auto \| jsonl \| csv) | no |  |
| `default_checker` (object/string/null) | no | Used for rows without checker and without expected answer. |

## `benchmark_import`

Import IFMTBench (instruction-following translation) as your suite. Importar benchmark de traducción.

Downloads the data from a pinned commit into the data folder, checks its SHA-256 and builds a seeded stratified sample (default 30 per constraint type plus 30 multi-constraint). Glossary rule, layout, structured data and code/tag checks are exact; style, context and a glossary the rule rejects use the judge model (settings judge.contestant) and stay pending without one. Data CC BY 4.0, scoring ported from the benchmark's Apache-2.0 code.
Sinónimos: traducción, glosario, terminología, formato, benchmark externo, seguir instrucciones al traducir, estilo, etiquetas, código

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `source` (string) | no | ifmtbench: instruction-following translation (glossary, style, context, layout, structured data, code and tags), CC BY 4.0 data. |
| `name` (string) | no | Name of the new suite (default IFMTBench). It must not exist yet. |
| `per_type` (integer) | no | Cases taken from each single-constraint type (6 types). 0: none. The whole file holds up to 1,644 per type. |
| `multi` (integer) | no | Cases taken from the multi-constraint file, spread evenly over its 5 combinations. 0: none. |
| `seed` (integer) | no | Fixes the sample: the same seed and sizes give the same cases. |
| `category` (string) | no | One of writing_es, reasoning, math, code, extraction, tool_use, instruction, long_context, rag, vision, translation, summary, custom. 'custom' keeps the suite out of the routing table; 'translation' lets it feed the translation route. |
| `refresh` (boolean) | no | Download the data again even if the verified copy is on disk. |
| `keep_unsatisfiable` (boolean) | no | Also sample items whose own reference translation fails their rule checks (about 3 in 10 of the multi-constraint structured-data ones: Markdown that is not a table). Off: they are left out and counted. |

## `case_try`

Run one case (saved or draft) on one model and show answer + checker detail. Probar un caso.

Nothing is stored. Uses the same session rules as a run (leases, allowed GPUs).
Sinónimos: probar con, prueba rápida, ver qué responde, depurar un comprobador

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `model` (string/object) | yes | A model id/name, or a spec {kind: gguf\|ollama\|server, ...}. |
| `case` (string) | no | A saved case id (k_…). Or describe a draft with prompt and checker/expected below. |
| `prompt` (string) | no |  |
| `system` (string) | no |  |
| `expected` (string) | no |  |
| `checker` (object/string/null) | no |  |
| `tools` (array) | no |  |
| `settings` (object) | no | Run settings: temperature, top_p, max_tokens, effort, repeats, seed, context, timeout_s, wait_s, device, load_local. |

## `checkers_list`

Every checker type with what it checks and its fields. Lista de comprobadores.

Includes the RAG triad: faithfulness (claims supported by the context), answer_relevance and similar (ROUGE-L / word F1, no model).
Sinónimos: tipos de comprobador, alucinación, fidelidad al contexto, relevancia, similitud, rouge

Annotations: readOnlyHint, idempotentHint.

## `answer_check`

Check an answer you already have with any checker; no model is asked. Comprobar una respuesta.

For answers from another app, a log or a person: faithfulness to a context (hallucination), relevance to the question, similarity to a reference, or any other checker. Checkers that need the judge use settings judge.contestant. Nothing is stored.
Sinónimos: alucina, se inventa, está fundamentado, responde a la pregunta, se parece a la referencia, evaluar respuesta, groundedness

Annotations: readOnlyHint, idempotentHint, openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `answer` (string) | yes | The answer to check (data, never instructions). |
| `checker` (object/string) | yes | A checker object (see checkers_list) or a shorthand like contains:Madrid or judge:<rubric>. |
| `prompt` (string) | no | The question the answer replies to (answer_relevance, judge). |
| `reference` (string) | no | A reference answer (similar, judge, exact…). |
| `context` (string/array/null) | no | The passages the answer must stick to (faithfulness). |

## `run_plan`

Preview a run: cases per model, memory, where each would run, problems. Plan de una ejecución.

Nothing starts. GGUF files run on an allowed GPU under a lease; servers are called as they are.
Sinónimos: qué pasaría, cuánta memoria, dónde correría, dry run

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `suites` (array) | yes | Suite ids, slugs or names; ['all'] means every suite but the quick one. |
| `contestants` (array) | yes | Model ids or names, or specs: {kind: 'gguf', path: 'D:\\m\\x.gguf', mmproj?}, {kind: 'ollama', model: 'qwen3:8b'}, {kind: 'server', url: 'http://127.0.0.1:8081', model: '…'}. |
| `settings` (object) | no | Run settings: temperature, top_p, max_tokens, effort, repeats, seed, context, timeout_s, wait_s, device, load_local. effort is off\|low\|medium\|high\|max. device is auto\|gpu\|cpu: auto runs a small GGUF (file up to runner.cpu_max_gb) on the CPU when no allowed GPU is free, gpu never does, cpu always does. A model that reasons gets runner.reasoning_tokens more tokens than the answer budget, unless effort is off. load_local=false makes the run refuse to start llama-server or load a model into Ollama (it only uses what is already served). |

## `run_start`

Start a run: suites x models (ids, names or specs of a file/server). Lanzar una medición.

Queued on the GPU lane; returns the run id (use wait_s to wait). Another app can pass {kind:'gguf', path} to evaluate a file it just made.
Sinónimos: medir, evaluar, benchmark, probar modelo, ejecutar pruebas, comparar base y afinado

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `suites` (array) | yes | Suite ids, slugs or names; ['all'] means every suite but the quick one. |
| `contestants` (array) | yes | Model ids or names, or specs: {kind: 'gguf', path: 'D:\\m\\x.gguf', mmproj?}, {kind: 'ollama', model: 'qwen3:8b'}, {kind: 'server', url: 'http://127.0.0.1:8081', model: '…'}. |
| `settings` (object) | no | Run settings: temperature, top_p, max_tokens, effort, repeats, seed, context, timeout_s, wait_s, device, load_local. effort is off\|low\|medium\|high\|max. device is auto\|gpu\|cpu: auto runs a small GGUF (file up to runner.cpu_max_gb) on the CPU when no allowed GPU is free, gpu never does, cpu always does. A model that reasons gets runner.reasoning_tokens more tokens than the answer budget, unless effort is off. load_local=false makes the run refuse to start llama-server or load a model into Ollama (it only uses what is already served). |
| `label` (string) | no |  |
| `wait_s` (number) | no | Wait this long for the run to finish before answering (0: return at once with the run id). |

## `run_status`

Progress of a run: per model state, cases done, memory, errors. Estado de una ejecución.

Sinónimos: cómo va, progreso, ejecución en curso

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | no | Run id (r_…). Empty: the running one, else the latest. |

## `run_cancel`

Cancel a run: the request in flight is dropped, results kept, shared servers untouched. Cancelar ejecución.

Only a llama-server that Galton started itself for a GGUF file is stopped. A server somebody else runs (llama-server, Ollama, an endpoint) is never stopped, killed or unloaded.
Sinónimos: parar, detener medición

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | no | Run id (r_…). Empty: the running one, else the latest. |

## `run_discard`

Discard a run's results from all statistics and routes (confirm=true). Descartar una ejecución.

The run stays in the history with the reason and a discarded badge. Use it when a run measured the wrong thing, for example when the model behind a shared server changed during it. run_restore undoes it.
Sinónimos: anular resultados, invalidar medición, resultados erróneos, ignorar ejecución

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes | Run id (r_…) of a run that has finished, failed or been cancelled. |
| `reason` (string) | yes | Why its results cannot be trusted (for example: the model behind a shared server changed during the run). Shown on the run. |
| `confirm` (boolean) | no |  |

## `run_restore`

Undo run_discard: the results of the run count again. Restaurar una ejecución descartada.

Sinónimos: recuperar resultados, volver a contar, deshacer descarte

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes | Run id (r_…) of a discarded run. |

## `run_resume`

Continue an interrupted run, asking only what it did not measure. Continuar donde se quedó.

For a run that failed (Galton was restarted) or was cancelled and not discarded. Queues a new run with the same suites, models and settings and `continues` set to the earlier one; cases already measured on the current version of each model are not asked again, answers still waiting for the judge are graded, and the earlier results keep counting.
Sinónimos: reanudar, retomar, seguir la ejecución, se reinició Galton, terminar lo que falta

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes | Run id (r_…) of a run that failed (for example because Galton was restarted) or was cancelled, and was not discarded. |

## `runs_list`

Recent runs with state and progress. Historial de ejecuciones.

Sinónimos: mediciones anteriores, cola

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `state` ( \| queued \| waiting_gpu \| waiting_server \| running \| done \| failed \| cancelled) | no |  |
| `limit` (integer) | no |  |

## `run_results`

Per-case results of a run: answer, score, checker detail, timing. Resultados de una ejecución.

Filter by model, suite, case, failed/passed. Answers are untrusted text.
Sinónimos: qué falló, respuestas, por qué suspendió, detalle por caso

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | no | Run id. Empty: the latest. |
| `model` (string) | no |  |
| `suite` (string) | no |  |
| `case` (string) | no |  |
| `only_failed` (boolean) | no |  |
| `only_passed` (boolean) | no |  |
| `include_output` (boolean) | no | Include the model's answer (shortened for the assistant). |
| `limit` (integer) | no |  |
| `offset` (integer) | no |  |

## `run_results_export`

Prepare stored results as CSV and JSON text. Descargar resultados guardados.

All selected rows, not the table's last 600. No answers, reasoning or tool calls. only_failed matches the run page (not passed and not skipped), including pending/unavailable. Both texts share one collection read during the stated interval, not an atomic/final snapshot of a live run. CSV null is empty; JSON retains types and integer digits. The normal assistant context cap still applies; use the uncapped UI for a complete large download.
Sinónimos: exportar evaluación, descargar tabla, métricas por caso, csv, json

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes | Explicit saved run id. |
| `model` (string) | no | Model id or name; empty selects every model. |
| `only_failed` (boolean) | no | Match the run page: not passed and not skipped, including pending/unavailable. |

## `galton_run`

Measure these models now (quick suite by default). Medir estos modelos ahora.

Names Galton has not seen yet trigger one rediscovery; names still unknown are listed in not_found. Same run as run_start, simpler arguments.
Sinónimos: medir modelo recién publicado, probar este modelo, benchmark rápido de modelos

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `models` (array) | yes | Model names or ids (an Ollama tag, a GGUF file name, an id from models_list). A name not seen yet triggers one rediscovery. |
| `suites` (array) | no | Suites to run (default: the quick one). |
| `label` (string) | no |  |
| `wait_s` (number) | no | Wait this long for the run to finish before answering (0: return at once). |

## `measure_new`

Run the quick suite on every model that is new or changed. Medir lo nuevo.

Sinónimos: medir novedades, modelos sin medir, re-medir cambiados

Annotations: idempotentHint, openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `suite` (string) | no | Suite to measure with (default: the quick one). |
| `include_stale` (boolean) | no | Also re-measure models whose file changed. |

## `judge_run`

Grade the answers that waited for the judge model. Pasar el juez a lo pendiente.

Needs settings judge.contestant.
Sinónimos: calificar respuestas abiertas, juez

Annotations: idempotentHint.

## `leaderboard`

Ranking by category or suite with 95 % intervals, speed and memory. Clasificación de modelos.

Ranked by the lower bound of the interval; stale results are not ranked; fits_16gb tells whether it fits one 16 GB GPU; truncated counts answers cut off by the token budget and warnings say when that makes a score understate the model.
Sinónimos: mejor modelo, ranking, tabla, puntuaciones, qué modelo es mejor en código

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `category` (string) | no | A task (general, writing_es, code, extraction, tool_use, long_context, rag, vision, summary, translation, math) or a suite category (writing_es, reasoning, math, code, extraction, tool_use, instruction, long_context, rag, vision, translation, summary, custom). Empty: everything. |
| `suite` (string) | no | One suite instead of a category. |
| `days` (integer) | no | Only results of the last N days (0: all). |
| `include_stale` (boolean) | no |  |
| `include_disabled` (boolean) | no |  |
| `limit` (integer) | no |  |

## `compare`

Paired comparison of two models: wins/losses, McNemar, bootstrap, verdict. Comparar dos modelos.

Verdict better / worse / no clear difference with p-value and n; a warning under 20 shared cases.
Sinónimos: es mejor, mejora la nueva cuantización, diferencia significativa, A contra B

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `a` (string) | yes | First model (id, name or alias). |
| `b` (string) | yes | Second model. |
| `category` (string) | no |  |
| `suite` (string) | no |  |
| `include_stale` (boolean) | no |  |
| `max_cases` (integer) | no | How many of the most different cases to list. |

## `gates_correlation`

Do the checks of a case fail together? phi, lift, kappa, redundant gates. Correlación entre comprobaciones.

Per pair of gates: phi, lift, kappa, P(A fails|B fails) with bootstrap 95 % intervals and a minimum-sample guard; findings in Spanish and English and which gate to drop.
Sinónimos: comprobaciones redundantes, fallos correlacionados, puertas, verificadores, juez redundante, kappa, phi

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `runs` (array) | no | Run ids to analyse together (they should share cases). Empty: every run that still counts. |
| `category` (string) | no | A task (general, writing_es, code, extraction, tool_use, long_context, rag, vision, summary, translation, math) or a suite category (writing_es, reasoning, math, code, extraction, tool_use, instruction, long_context, rag, vision, translation, summary, custom). Empty: everything. |
| `suite` (string) | no | One suite instead of a category. |
| `include_stale` (boolean) | no |  |
| `min_samples` (integer) | no | Answers a pair of gates must share before any conclusion is drawn about it. |
| `strong_phi` (number) | no | phi from which two gates count as carrying the same signal (the interval must also stay above 0.5). |
| `resamples` (integer) | no | Bootstrap resamples for the intervals. |

## `recommend`

Best measured model for a task described in words, with why and runner-up. Qué modelo usar.

Guesses the category from keywords (es/en) or takes category; says so when there is not enough evidence.
Sinónimos: recomienda un modelo, cuál uso para, el mejor para traducir

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `task` (string) | yes | What the model will be used for, in plain words (any language). |
| `category` (string) | no | Force a task category: general, writing_es, code, extraction, tool_use, long_context, rag, vision, summary, translation, math. |

## `routes_get`

The routing table: winner per task, what changes if published now. Tabla de rutas.

Other Hoard apps read the published file through Hoard Link.
Sinónimos: rutas, qué modelo usa cada app, diferencias con lo publicado

Annotations: readOnlyHint, idempotentHint.

## `routes_publish`

Publish the routing table to routes.json for Hoard Link. Publicar rutas.

Atomic write; categories with too few checked cases are left out; emits galton.routes.updated.
Sinónimos: aplicar, publicar tabla, actualizar rutas

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `note` (string) | no |  |

## `arena_next`

A blind pair of answers to the same prompt to vote on. Siguiente par de la arena.

Sinónimos: comparar a ciegas, votar, preferencia humana

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `category` (string) | no |  |

## `arena_vote`

Vote a|b|tie|both_bad on an arena pair. Votar en la arena.

Sinónimos: elegir respuesta, empate, ambas malas

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `pair` (string) | yes | Pair id from arena_next. |
| `vote` (a \| b \| tie \| both_bad) | yes |  |

## `arena_ratings`

Bradley-Terry ratings from the arena votes, per category. Puntuaciones de la arena.

Sinónimos: elo, ranking humano

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `category` (string) | no |  |

## `settings_get`

All settings with values, types and defaults, plus the GPU state. Ver ajustes.

Sinónimos: configuración, GPUs permitidas, política de rutas, juez, horas de silencio

Annotations: readOnlyHint, idempotentHint.

## `settings_set`

Change settings: allowed GPUs, llama-server path, judge, runner, watch, routes policy. Cambiar ajustes.

Allowing a GPU reserved for the owner needs confirm_reserved=true and an explicit request from the user.
Sinónimos: permitir GPU, ruta de llama-server, política, horas de silencio, ejecución de código

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `values` (object) | yes | Setting key -> value. See settings_get for the keys and types. |
| `merge` (boolean) | no | For object settings (routes.policy): change only the fields given and keep the others as they are now. Without it the fields not given go back to their defaults. |
| `confirm_reserved` (boolean) | no | Needed to allow a GPU that is reserved for the owner of this computer: only if the user explicitly said so. |

## `gpu_status`

Per GPU: total/used/free, allowed or reserved, leases and queue. Estado de las GPU.

Sinónimos: memoria de vídeo, VRAM libre, quién usa la GPU, reservas

Annotations: readOnlyHint, idempotentHint.

## `notices_list`

Notices: regressions, improvements, new or changed models, failed runs. Avisos.

Sinónimos: regresión, novedades, qué ha pasado

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `unseen` (boolean) | no |  |
| `kind` (string) | no | regression, improvement, new_model, changed_model, run_failed… |
| `limit` (integer) | no |  |
| `mark_seen` (boolean) | no | Mark the listed notices as seen. |

## `housekeeping_run`

Tidy caches, unused images and old unvoted arena pairs; stop leftover servers. Mantenimiento.

Sinónimos: limpiar, ordenar

Annotations: idempotentHint.

## `rulebench_tasks`

Rule bench tasks and the exact prompts for variant A/B/C and a pressure. Tareas del banco de reglas.

Twenty small repositories (bugs, small features, refactor traps with dead code and other people's comments) with hidden tests and a reference patch.
Sinónimos: reglas de código, prompt de sistema, variantes, alcance de cambios, presión

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `task_id` (string) | no | One task: its system and user prompt for the variant and pressure. Empty: the list. |
| `variant` (A \| B \| C) | no | A no rules, B the four code rules in their original wording, C positive and short. |
| `pressure` (neutral \| favor \| contra) | no | contra adds 'do it fast and tidy up the file' to the request. |

## `rulebench_score`

Score an answer to a rule-bench task: hidden tests, lines beyond the reference, lost comments. Puntuar.

Applies the answer's file blocks in a scratch folder and runs the hidden tests (needs checks.allow_code_execution). Also reports test-first order.
Sinónimos: evaluar cambio, líneas fuera de alcance, código muerto borrado, comentarios ajenos, test antes que código

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `task_id` (string) | yes |  |
| `answer` (string) | yes | The model's answer: one ```file:<path> block per changed file. |
| `variant` (string) | no | Label stored with the sample. |
| `pressure` (string) | no | Label stored with the sample. |

## `harness_catalog`

Harnesses, variants, tasks and tools of the harness bench. Catálogo de arneses y tareas.

Four base harnesses (native_tools, json_text, xml_text, react_text), the .long and .trim variants, command-line harnesses (off by default), the 24 base tasks with their kinds and the language bench (27 tasks in es and en, pass langs to run them).
Sinónimos: arnés, bucle de agente, tareas con herramientas, function calling, ReAct, XML, JSON

Annotations: readOnlyHint, idempotentHint.

## `harness_matrix_start`

Run a model x harness x task matrix: model or harness at fault. Lanzar matriz de arneses.

Same model under each harness, same token cap and turn limit in every cell. Reports pass@1, format failures apart from wrong answers, needless tool calls, turns, tokens; a trial whose verifier never ran is None, not 0. With langs [es, en] it runs the language-bench tasks and adds the language axis (language consistency, empty searches, calls against English, arguments in another language than the data). Queued on the GPU lane; use wait_s to wait, dry_run to count the trials only.
Sinónimos: medir arneses, mismo modelo distinto arnés, fallo de formato, function calling nativo, ReAct, evaluar agente

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `models` (array) | yes | Model ids or names, or specs {kind: 'server', url, model} / {kind: 'ollama', model} / {kind: 'gguf', path}. |
| `harnesses` (array) | no | Harness ids: native_tools, json_text, xml_text, react_text, each with optional .long (long system prompt) and/or .trim (older observations replaced by a note), or cli:<name> for a configured command-line harness. Empty: the four base harnesses. |
| `tasks` (array) | no | Task ids (see harness_catalog). Empty: all 24 base tasks, all 27 language-bench tasks when `langs` is set, or those of `kinds`. |
| `kinds` (array) | no | Only tasks of these kinds. Base: calc, convert, docs, files, chain, notool, recovery. Language bench: lookup, filter, aggregate, cart, recovery, write, notool. |
| `langs` (array) | no | Run the language-bench tasks in these languages (es, en): the same 27 tasks with the same ids, tools and enum values, localised. Adds the language axis to the matrix: one column per harness and language, with language consistency, empty searches, tool calls against the English run and tool arguments in another language than the data. Empty: the base tasks. |
| `k` (integer) | no | Samples per task and cell. With k above 1 the temperature defaults to 0.7 and pass@k and pass^k are reported. |
| `settings` (object) | no | The same for every cell: max_tokens (per reply, default 1024), max_turns (default 8), max_format_errors (default 3), plus the run settings temperature, top_p, max_tokens, effort, seed, context, timeout_s, wait_s, device, load_local. |
| `label` (string) | no |  |
| `wait_s` (number) | no | Wait this long for the matrix to finish before answering (0: return at once with its id). |
| `dry_run` (boolean) | no | Only count the trials and show the settings every cell will share; start nothing. |

## `harness_matrix_status`

Progress of a harness matrix: per model state, trials done, errors. Estado de la matriz.

Sinónimos: cómo va la matriz de arneses, progreso

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `matrix` (string) | no | Matrix id (m_…). Empty: the running one, else the latest. |

## `harness_matrix_result`

Harness matrix as a table with a short conclusion: harness vs model. Resultado de la matriz de arneses.

Per cell: pass@1 with interval, format failure rate, wrong-answer rate, extra tool calls, turns, tokens, None count. Conclusion: whether the difference between harnesses is larger or smaller than between models, and the best harness per model. In a language matrix the columns are harness@lang and a language block compares each language with English.
Sinónimos: tabla modelo por arnés, mapa de calor, mejor arnés por modelo, qué falla, el modelo o el arnés

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `matrix` (string) | no | Matrix id (m_…). Empty: the running one, else the latest. |

## `harness_cell`

One matrix cell: its tasks and, for one task, the traces and the reason of each failure. Detalle de una celda.

Every turn with the tool calls, results, refused calls and format errors.
Sinónimos: trazas, por qué falló el formato, llamadas sobrantes, depurar un arnés

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `matrix` (string) | no | Matrix id. Empty: the running one, else the latest. |
| `model` (string) | yes | Model name or id as shown in the matrix. |
| `harness` (string) | yes | Harness id of the column, for example json_text.long; in a language matrix add the language, for example json_text.long@es. |
| `task` (string) | no | A task id: adds the full traces (every turn, call, result and format failure) of its trials. |

## `harness_matrix_cancel`

Cancel a harness matrix: results kept, shared servers untouched. Cancelar matriz de arneses.

Sinónimos: parar matriz, detener

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `matrix` (string) | no | Matrix id (m_…). Empty: the running one, else the latest. |

## `harness_matrix_list`

Recent harness matrices with state and conclusion. Historial de matrices de arneses.

Sinónimos: matrices anteriores, mediciones de arneses

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `limit` (integer) | no |  |

## `langbench_task`

Language bench: the same tool tasks in Spanish and English, side by side. Tareas del banco de idiomas.

Lists the 27 tasks (kind, domain) or, for one id, both versions: prompt, tools with their descriptions, oracle calls and final answer. canonical adds the entity map and the type of each string: Preserve (ids, tool names, enums), Translate-Free (instructions) and Translate-Canonical (entities shared by prompts, data and checks).
Sinónimos: tareas es en, localización, mapa canónico, tools en castellano, idioma de las herramientas

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `task` (string) | no | A language-bench task id. Empty: list every task with its kind and domain. |
| `canonical` (boolean) | no | Also return the canonical entity map (the Translate-Canonical strings) and the type of each string. |

## `langbench_judge`

Judge the language of a reply, or of a tool argument against the data. Juzgar el idioma de una respuesta.

Local lexical judge: it looks only at the prose (ids, numbers, prices, brands and entity names are ignored) and returns the share of language-marked words in the expected language; with argument=true it says whether a search argument is in the language of the data. No model is called.
Sinónimos: consistencia de idioma, responde en otro idioma, argumento en inglés, juez local

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `text` (string) | yes | A reply to judge (prose only: ids, numbers, prices and entity names are ignored) or a tool argument. |
| `lang` (es \| en) | yes | The language the reply was expected in, or the language of the data for a tool argument. |
| `argument` (boolean) | no | Treat the text as a tool argument and say whether it is in the language of the data (lang) instead of judging a reply. |

## `videobench_catalog`

Video bank: briefs for animations written as code, with their author and post. Catálogo del banco de vídeo.

Public prompts people shared for animations made with code (HTML, canvas, SVG, WebGL): those with the whole prompt published and more than 400 characters. Filter by category, technique or standalone (no attached file, link or image). The list is MIT; every video belongs to its author.
Sinónimos: prompts de animación, vídeos hechos con código, motion, explainer, 3d, qué pedir a un modelo

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `category` (string) | no | motion, 3d, interactive or explainer. Empty: all. |
| `tag` (string) | no | A technique tag such as canvas, svg, threejs, shader, css, particles. Empty: all. |
| `standalone` (boolean/null) | no | true: only briefs that need no attached file, link or image (the ones a run picks by default); false: only the others. Omit: all. |
| `query` (string) | no | Words that must appear in the brief or its id. |
| `limit` (integer) | no |  |
| `offset` (integer) | no |  |

## `videobench_prompt`

One video brief: the whole prompt with its author and post. Un encargo de vídeo completo.

Sinónimos: ver el prompt, autor, post original

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `prompt` (string) | yes | A brief id (see videobench_catalog). Returns the whole prompt with its author and post. |

## `videobench_start`

Video bank: a model writes animations as code, a jury scores the renders. Lanzar banco de vídeo.

Per brief: the model writes a self-contained HTML page with window.seek(t) (the HTML-animation contract of the video editor); a headless browser renders 12 frames and a contact sheet; automatic checks say whether it renders, moves, is deterministic and stays offline; a second session scores the frames on the 7-dimension rubric (sound is not scored: the animation is silent). The jury is a model: its scores are a judgement. Runs in the background; dry_run only shows the plan.
Sinónimos: medir un modelo haciendo animaciones, HTML motion, jurado, rúbrica, calidad de vídeo con código

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `model` (string/object) | yes | The model that writes the animations: id or name, or a spec {kind: 'server', url, model}. |
| `jury` (object/string/null) | no | The model that scores the renders, in a fresh session. Empty: the same model (it tends to be kinder to its own pages; the result says so). A model that cannot see images judges from the automatic checks and the text of the page. |
| `prompts` (array) | no | Brief ids. Empty: a sample chosen by count, seed and the filters. |
| `count` (integer) | no | How many briefs to sample when `prompts` is empty (the same seed gives the same briefs). |
| `seed` (integer) | no |  |
| `category` (string) | no |  |
| `tag` (string) | no |  |
| `standalone` (boolean/null) | no | Sample only briefs that need no attached file, link or image. null: any. |
| `settings` (object) | no | max_tokens (default 16000), timeout_s (900), temperature (0.7), frames (12), duration_cap (30 s), default_duration (10 s), render_timeout_s (240), jury_images (auto\|true\|false), jury_max_tokens (4000), plus the run settings seed, effort, device. |
| `label` (string) | no |  |
| `wait_s` (number) | no | Wait this long for the run to finish before answering (0: return at once with its id). |
| `dry_run` (boolean) | no | Only show which briefs would run, how the jury would judge and whether a browser is available; start nothing. |

## `videobench_status`

Progress of a video-bank run: phase, briefs written, rendered and judged. Estado del banco de vídeo.

Sinónimos: cómo va, progreso del vídeo

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | no | Run id (vb_…). Empty: the running one, else the latest. |

## `videobench_result`

Video-bank run as a table: per brief, renders, moves, jury scores. Resultado del banco de vídeo.

Summary: render rate, usable rate (renders, moves, deterministic, no page errors), mean score, per dimension and per category. Items that never reached the jury are None, not zero, and are left out of the means.
Sinónimos: tabla de resultados, notas del jurado, qué modelo hace mejores animaciones

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | no | Run id (vb_…). Empty: the running one, else the latest. |

## `videobench_item`

One video-bank item: the brief, automatic checks, jury scores, fixes and the files. Detalle de un encargo.

The jury's summary, critical issues and fixes; render measures (declared length, frames, errors, blocked requests); optionally the page source. The contact sheet and frames are served by the app at /api/videobench/<run>/<brief>/sheet.png.
Sinónimos: ver el vídeo, hoja de contactos, fallos del jurado, el HTML

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | no | Run id (vb_…). Empty: the running one, else the latest. |
| `prompt` (string) | yes | The brief id of the item. |
| `html` (boolean) | no | Include the page the model wrote (can be long). |

## `videobench_cancel`

Cancel a video-bank run: finished items kept. Cancelar banco de vídeo.

Sinónimos: parar, detener

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | no | Run id (vb_…). Empty: the running one, else the latest. |

## `videobench_list`

Recent video-bank runs with state and summary. Historial del banco de vídeo.

Sinónimos: ejecuciones anteriores de vídeo

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `limit` (integer) | no |  |

## `speechbench_engines`

Speech-to-text engines for the speech bench; add command engines. Motores de voz a texto.

faster-whisper:<size> runs in a Python that has faster-whisper (found among this app and the family's venvs, or set with whisper_python), CPU int8 by default, one warm model per run. A command engine (name + command with {audio}, {language}) covers any other recogniser, e.g. a streaming model's CLI; it prints text or JSON {text, first_text_s}. Without arguments it only lists.
Sinónimos: motores de reconocimiento de voz, añadir motor, faster-whisper, R2T2, transcriptor, STT

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `name` (string) | no | Command engine name (lower case); with command, saved as cmd:<name>. |
| `command` (array/null) | no | Program and arguments; {audio} is the file, {language} the language code. It prints plain text or JSON {text, first_text_s}. |
| `timeout_s` (number) | no |  |
| `cwd` (string) | no |  |
| `note` (string) | no |  |
| `delete` (boolean) | no | Remove the command engine called name. |
| `whisper_python` (string/null) | no | Python with faster-whisper for the faster-whisper engines ('' = find one). |

## `speechbench_corpus`

Check a speech corpus: audio files with their reference text. Comprobar un corpus de voz.

Pairs clip.wav + clip.txt (subfolders too) or corpus.jsonl {audio, text}. Returns files, total audio, problems (no or empty reference) and a preview. References must be written by a person, not by a recogniser.
Sinónimos: corpus de audio, referencias, transcripciones de referencia, grabaciones

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `folder` (string) | yes | Folder with audio files and a .txt reference of the same name each, or a corpus.jsonl. |

## `speechbench_start`

Speech bench: transcribe a corpus with engines; WER, CER and latency. Lanzar banco de voz a texto.

Every engine transcribes every file; each hypothesis is scored against its reference (corpus WER/CER = total errors over total reference words, with substitutions, deletions, insertions), and timed: model load, time to the first text, decode time and real-time factor. Runs in the background on the CPU; dry_run shows the plan.
Sinónimos: comparar reconocedores, tasa de error de palabras, latencia de transcripción, Whisper contra otro

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `folder` (string) | yes | The corpus folder (see speechbench_corpus). |
| `engines` (array) | yes | faster-whisper:<tiny\|base\|small\|medium\|large-v3\|large-v3-turbo> and/or cmd:<name>. |
| `language` (string/null) | no | Language code (es, en...); omit to let each engine detect it. |
| `accents` (boolean) | no | false folds accents before comparing (canción = cancion). |
| `device` (string) | no | For faster-whisper; cpu keeps the GPUs free. |
| `compute` (string) | no |  |
| `limit` (integer) | no | At most this many files of the corpus. |
| `name` (string) | no |  |
| `dry_run` (boolean) | no | Only check the corpus and engines and show the plan. |

## `speechbench_result`

A speech-bench run: state and per-engine WER, CER, latency, RTF. Resultado del banco de voz.

items=true adds every file with its reference, hypothesis, WER, CER, times and errors (engine filters).
Sinónimos: cómo va, resultado, errores por archivo, qué motor transcribe mejor

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes |  |
| `items` (boolean) | no | Include every file: reference, hypothesis, WER, CER, times, errors. |
| `engine` (string/null) | no | With items: only this engine. |

## `speechbench_list`

Recent speech-bench runs. Historial del banco de voz a texto.

Sinónimos: ejecuciones anteriores de voz

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `limit` (integer) | no |  |

## `speechbench_cancel`

Cancel a speech-bench run: files already scored are kept. Cancelar banco de voz.

Sinónimos: parar, detener

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes |  |
| `items` (boolean) | no | Include every file: reference, hypothesis, WER, CER, times, errors. |
| `engine` (string/null) | no | With items: only this engine. |

## `logit_judge_run`

Logit judge: rubric answered from next-token probabilities, base vs adapter. Juez por logits.

Instead of asking a judge to write JSON, each question (choice with labels A, B, C... or a 0-9 score) is sent with its context and the probability of every option token is read: chosen option, distribution, expected score. Backends: a Hugging Face model in the trainer environment (with an optional LoRA adapter: base and adapter are both scored) or an chat-completions server that returns logprobs (llama.cpp server, vLLM, Ollama). A server that ignores logprobs is reported, never replaced by a generated answer. Stored as a run; compare sets by their deltas.
Sinónimos: juez logit, probabilidades del siguiente token, logprobs, rúbrica, antes y después del adaptador, LoRA contra base, puntuación esperada

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `targets` (array) | yes | Who answers. A running chat-completions server that returns logprobs (a model id or name, or {kind: 'server', url, model}: llama.cpp server, vLLM, a recent Ollama) or a local Hugging Face model {kind: 'hf', base_path, adapter_path?}. An hf target with an adapter gives two result sets, the base model and the base model with the adapter, from one load. Deltas are always the later sets against the first. |
| `rubric` (array) | yes | Questions {id, type: choice\|score, instructions, options?, min?, max?, anchors?}. A choice lists its option texts (read as the labels A, B, C...). A score is a single digit scale, default 0..9 (min 0 or more, max 9 or less); anchors {"0": "text"} describe digits. |
| `items` (array) | yes | What is judged: shared contexts, each a text or {id, context}. Every question is asked about every context. |
| `label` (string) | no |  |
| `dry_run` (boolean) | no | Check the rubric, resolve the targets and show the labels and the number of questions; ask nothing. |
| `detail` (boolean) | no | Include every item's answers (chosen option, distribution, expected score, log-probabilities). Off: only the totals per question and the deltas. |

## `logit_judge_result`

A logit-judge run: totals, distributions and base-vs-adapter deltas. Resultado del juez logit.

Sinónimos: ver el juez logit, deltas, antes y después, distribución de probabilidad

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | no | Run id (lj_…). Empty: the latest logit-judge run. |
| `detail` (boolean) | no | Include every item's answers, not only the totals per question and the deltas. |

## `logit_judge_list`

Recent logit-judge runs. Historial del juez por logits.

Sinónimos: ejecuciones anteriores del juez logit

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `limit` (integer) | no |  |

## `logit_judge_probe`

Check whether a server returns logprobs. Servidor con logprobs.

Asks one two-option question and reports whether the server sent the alternatives of the first token, unreachable, or ignored the request.
Sinónimos: comprobar logprobs, llama-server, vLLM, Ollama, top_logprobs

Annotations: readOnlyHint, idempotentHint, openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `targets` (array) | yes | Servers to test: a model id or name, or {kind: 'server', url, model}. |

## `logit_judge_cancel`

Stop a running logit-judge run (kills the model worker). Cancelar el juez por logits.

Sinónimos: parar, detener

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | no | Run id (lj_…). Empty: every running logit-judge run. |

## `logit_judge_delete`

Delete a logit-judge run for good (confirm=true). Borrar una ejecución del juez por logits.

Removes the run and its stored distributions from the history, runs_list and comparisons; a run still working is refused (cancel it first). run_discard only hides a suite run's results from the statistics; this removes a logit-judge run entirely.
Sinónimos: borrar, eliminar, quitar ejecución del juez logit

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes | Run id (lj_…) of a logit-judge run that has finished, failed or been cancelled. |
| `confirm` (boolean) | no |  |

## `agentbench_catalog`

Agent bench: benchmarks, their task subsets and the configured agent harnesses. Catálogo del banco de agentes.

Published agentic benchmarks (terminal work, repository fixes, multi-language edits, dialogues with tools, algorithmic problems), the fixed subset Galton uses of each with its dev and test split, and the harness lines of the setting agentbench.harnesses.
Sinónimos: benchmarks de agentes, arneses reales, terminal, swe, polyglot, tau, livecodebench

Annotations: readOnlyHint, idempotentHint.

## `agentbench_start`

Run real agent harnesses on published benchmarks with one model. Lanzar campaña de agentes.

Each task runs in its own container: the harness is installed, works on the task, then the task's hidden tests decide. Same tasks, same model, same limits for every harness. Trials whose verifier never ran are None, not 0, and are retried. Runs in the background; dry_run shows the subsets.
Sinónimos: comparar arneses, qué agente es mejor con este modelo, campaña, benchmark de agentes

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `benchmarks` (array) | no | Benchmark ids from agentbench_catalog (tb2, swev, polyglot, tau3, lcb). Empty: all. |
| `harnesses` (array) | no | Harness names from the setting agentbench.harnesses. Empty: every line that is switched on. |
| `model` (string) | no | Model id the server knows. Empty: the setting agentbench.model. |
| `base_url` (string) | no | Address of the server that serves it (common chat-completions protocol, ending in /v1). Empty: the setting agentbench.base_url. |
| `k` (integer) | no | Attempts per task and harness. |
| `splits` (array) | no | dev: the tasks a harness may be tuned on; test: only measured. |
| `label` (string) | no |  |
| `slots` (integer) | no | Trials in flight at once (0: the setting). |
| `adopt` (string) | no | Name of a campaign whose jobs another process runs: only read its job folders, launch nothing. |
| `dry_run` (boolean) | no | Only show the task subsets of each benchmark; start nothing. |

## `agentbench_status`

Progress of an agent-bench campaign. Estado de la campaña de agentes.

Sinónimos: cómo va la campaña, progreso del banco de agentes

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `campaign` (string) | no | Campaign id or name. Empty: the running one, else the latest. |

## `agentbench_result`

Agent-bench board: pass rate per benchmark and harness, with ranking. Resultado de la campaña.

Per cell: verified trials, pass rate with its 95 % interval, None trials apart, failure reasons, tokens, time. Per benchmark: ranking, the focus harness's paired wins and losses against each other harness with an exact sign test.
Sinónimos: clasificación de arneses, quién gana, mapa de calor, comparación por tarea

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `campaign` (string) | no | Campaign id or name. Empty: the running one, else the latest. |
| `split` ( \| dev \| test) | no | Only the trials of that split (empty: all). |
| `focus` (string) | no | Harness compared, task by task, against every other one. |

## `agentbench_cell`

One cell of a campaign: every trial of a harness on a benchmark. Detalle de una celda.

Sinónimos: qué tareas falló, por qué falló

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `campaign` (string) | no |  |
| `benchmark` (string) | yes |  |
| `harness` (string) | yes |  |

## `agentbench_trial`

One trial with the logs the harness and the verifier left. Detalle de una prueba con sus registros.

Sinónimos: log del agente, salida de los tests, depurar arnés

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `trial` (integer) | yes | Trial id from agentbench_cell. |
| `limit` (integer) | no | Characters kept from the end of each log file. |

## `agentbench_sync`

Read a campaign's job folders again into the database. Releer las carpetas de la campaña.

Sinónimos: actualizar resultados

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `campaign` (string) | no | Campaign id or name. Empty: the running one, else the latest. |

## `agentbench_cancel`

Cancel an agent-bench campaign: trials already finished are kept. Cancelar campaña de agentes.

Sinónimos: parar campaña

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `campaign` (string) | no | Campaign id or name. Empty: the running one, else the latest. |

## `agentbench_list`

Recent agent-bench campaigns. Historial de campañas de agentes.

Sinónimos: campañas anteriores

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `limit` (integer) | no |  |

## `pygmalion_suit_eval_define`

Define a suit evaluation: cases with executable checks and a candidate. Definir evaluación de traje.

Three arms are compared: generic tools, the current suit and a candidate improvement. Cases the candidate was built from (used_for_building, candidate.built_from_cases) are excluded from the score.
Sinónimos: evaluar un traje, casos de aceptación, brazos, mejora candidata, comprobaciones ejecutables

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `id` (string) | no | Empty: a new evaluation. An existing id replaces its definition (cases with results cannot be removed). |
| `suit` (string) | yes | Slug of the suit in Darwin's Hoard, e.g. repository-intake. |
| `name` (string) | no |  |
| `cases` (array) | yes |  |
| `candidate` (null) | no |  |
| `notes` (string) | no |  |

## `pygmalion_suit_library`

Suit library: Darwin's suits with their evaluations and decision. Biblioteca de trajes.

One suit per kind of engagement, read from Darwin's Hoard (suiteval.darwin_url); each with the evaluations kept here and the decision of the latest: adopt, keep, tradeoff or insufficient. Without Darwin, the suits known from local evaluations.
Sinónimos: biblioteca de trajes, qué encargos sabes resolver, trajes por tipo de encargo, adoptar mejora

Annotations: readOnlyHint, idempotentHint, openWorldHint.

## `pygmalion_suit_eval_list`

List suit evaluations with cases, scored cases and export state. Evaluaciones de trajes.

Sinónimos: evaluaciones de un traje, historial

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `suit` (string) | no | Only the evaluations of this suit. |

## `pygmalion_suit_eval_get`

One suit evaluation: definition, records and score per arm with warnings. Detalle de una evaluación.

Sinónimos: ver evaluación de traje, resultados por caso

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `id` (string) | yes | Evaluation id (se_…). |

## `pygmalion_suit_eval_record`

Record what an arm produced for a case; checks run on it. Registrar resultado de un caso.

Runs the case's checks on the deliverable folder, answer text, numbers and judge verdicts given. Calling again for the same case, arm and run updates it, so a person can add correction_minutes later.
Sinónimos: registrar un caso, minutos de corrección, tiempo total, defectos, tokens

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `id` (string) | yes | Evaluation id. |
| `case` (string) | yes |  |
| `arm` (generic \| current \| candidate) | yes | generic: plain tools. current: the suit as it is. candidate: the improvement under test. |
| `run` (integer) | no | Repetition number when the same case and arm are run more than once (averaged per case). |
| `deliverable` (string) | no | Folder with what the arm produced; the file, JSON and number checks read it (never outside it). |
| `answer` (string/null) | no | The arm's text answer, for regex checks with target answer. |
| `values` (object/null) | no | Numbers for numeric checks that use value: <name>. |
| `judgements` (object/null) | no | Verdicts of judge checks by check id: {ok: bool, score?, judge?, note?}. From a judge model, the logit judge or a person. |
| `correction_minutes` (number/null) | no | Minutes a person spent correcting the deliverable. Entered by a person; leave out when unknown. |
| `total_time_s` (number/null) | no | Total time of the run in seconds. |
| `defects` (integer/null) | no | Defects found. Left out: counted from the failed checks. |
| `tokens` (integer/null) | no | Tokens used, when the harness reports them. |
| `notes` (string/null) | no |  |

## `pygmalion_suit_eval_check`

Try acceptance checks on a folder or text without saving. Probar comprobaciones.

Sinónimos: probar un criterio de aceptación, esquema JSON, tolerancia numérica, expresión regular

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `checks` (array) | yes | Same shape as the checks of a case. |
| `deliverable` (string) | no |  |
| `answer` (string) | no |  |
| `values` (object/null) | no |  |
| `judgements` (object/null) | no |  |

## `pygmalion_suit_eval_score`

Score a suit evaluation per arm, with spread; build cases excluded. Puntuar evaluación.

Cases used to build the candidate are excluded and listed; paired comparisons give wins, losses and an exact sign test. verdict answers candidate_vs_current and current_vs_generic with adopt, keep, tradeoff or insufficient: accepted deliverables first, then correction minutes, defects and time; tokens are reported but never decide.
Sinónimos: agregado por brazo, dispersión, candidato frente a actual, casos excluidos

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `id` (string) | yes | Evaluation id (se_…). |

## `pygmalion_suit_eval_export`

Send a suit evaluation to Darwin's Hoard or write a JSON file. Exportar a Darwin.

Refused while no case outside the building set has a result. In Darwin the suit becomes evaluated.
Sinónimos: enviar resultados a Darwin, marcar traje como evaluado, fichero de importación

Annotations: idempotentHint, openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `id` (string) | yes |  |
| `mode` (auto \| api \| file) | no | auto: Darwin's API if it answers, otherwise a JSON file. api: only the API. file: only a file. |
| `url` (string) | no | Darwin's base URL; default the setting suiteval.darwin_url. |

## `pygmalion_suit_eval_delete`

Delete a suit evaluation and its results (confirm=true). Borrar evaluación de traje.

Sinónimos: quitar evaluación

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `id` (string) | yes |  |
| `confirm` (boolean) | no |  |

## `rag_sets`

RAG bench: case sets with their documents, cases by state and runs. Conjuntos de casos RAG.

A case set is a small corpus plus cases: a question, the passages that answer it and the expected answer. Start here to evaluate a RAG pipeline piece by piece (retriever, reranker, generator).
Sinónimos: evaluar un RAG, tubería RAG, recuperador, reordenador, generador, conjuntos de preguntas

Annotations: readOnlyHint, idempotentHint.

## `rag_set_save`

Create a RAG case set from documents, or add documents and cases to one. Crear conjunto de casos RAG.

Documents come as texts or as absolute paths of text files and folders. Cases can be written by hand here, or proposed by a model with rag_cases_generate.
Sinónimos: corpus de prueba, documentos, añadir casos, conjunto de evaluación RAG

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `id` (string) | no | Empty: a new set. An existing id (or name): documents and cases are added to it. |
| `name` (string) | no |  |
| `description` (string) | no |  |
| `documents` (array) | no | The corpus: texts or {name, text}. |
| `paths` (array) | no | Absolute paths of text files or folders (txt, md, csv, json, html...) to read as documents. |
| `cases` (array) | no |  |
| `replace_documents` (boolean) | no | Replace the documents of an existing set instead of adding to them. |

## `rag_set_get`

One RAG case set: documents, cases with their passages and its runs. Detalle de un conjunto RAG.

Sinónimos: ver casos, casos propuestos, pasajes de referencia

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `set` (string) | yes | Set id (rs_…) or name. |
| `status` ( \| proposed \| accepted \| rejected) | no | Only the cases in this state. |
| `with_text` (boolean) | no | Include the full text of the documents. |

## `rag_set_delete`

Delete a RAG case set with its cases and runs (confirm=true). Borrar conjunto de casos RAG.

Sinónimos: quitar conjunto, borrar corpus de prueba

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `set` (string) | yes |  |
| `confirm` (boolean) | no |  |

## `rag_cases_generate`

A model proposes RAG cases from the set's documents, for a person to review. Generar casos RAG.

For fragments spread over the documents the model writes a question, the short answer and the sentence that supports it; the quote is checked against the fragment and becomes the reference passage. Cases stay proposed until accepted (rag_cases_review).
Sinónimos: generar preguntas del corpus, casos sintéticos, no escribir el conjunto a mano, preguntas con su pasaje de origen

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `set` (string) | yes |  |
| `model` (string/object) | yes | The model that writes the cases: a running server model (id or name) or {kind: server, url, model}. |
| `n` (integer) | no | Answerable cases to propose. |
| `per_fragment` (integer) | no | Questions asked per document fragment. |
| `unanswerable` (integer) | no | Also propose this many questions the passage does not answer (to measure abstention). A person must check that no other document answers them. |
| `chunk_chars` (integer) | no | Size of the fragments the model reads. |
| `seed` (integer) | no | Which fragments are picked. |

## `rag_case_save`

Add a RAG case by hand or edit one: question, passages, expected answer, state. Guardar caso RAG.

With id, only the fields given change; status accepts or rejects it.
Sinónimos: editar caso, corregir pregunta, aceptar caso, pregunta sin respuesta

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `set` (string) | no | The set a new case goes into. |
| `id` (string) | no | Case id (rk_…) to change; empty adds a case. |
| `question` (string/null) | no |  |
| `expected` (string/null) | no |  |
| `passages` (array/null) | no |  |
| `answerable` (boolean/null) | no |  |
| `require` (string/null) | no |  |
| `status` (string/null) | no | Review state: only accepted cases are evaluated. |
| `notes` (string/null) | no |  |

## `rag_cases_review`

Accept or reject proposed RAG cases, some or all. Revisar casos RAG propuestos.

Sinónimos: aceptar todos, rechazar casos, revisión de casos generados

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `set` (string) | yes |  |
| `status` (proposed \| accepted \| rejected) | yes |  |
| `ids` (array) | no |  |
| `all_proposed` (boolean) | no | Apply to every proposed case of the set. |

## `rag_case_remove`

Delete one RAG case (confirm=true). Borrar un caso RAG.

Sinónimos: quitar caso

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `id` (string) | yes |  |
| `confirm` (boolean) | no |  |

## `rag_probe`

Ask a RAG pipeline one question and see its passages and answer; nothing stored. Probar tubería RAG.

To check an adapter (faustus, http) or a configuration before a run: what was retrieved, the order after the reranker, what the generator got and what it answered.
Sinónimos: probar adaptador, ver qué recupera, depurar tubería, una pregunta de prueba

Annotations: readOnlyHint, idempotentHint, openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `pipeline` (object) | yes | The pipeline to evaluate. {kind: builtin}: this bench's own plain pipeline over the documents of the case set: retriever (bm25 \| random \| none), chunk_chars and chunk_overlap (fragment size), depth (passages retrieved, default 10), reranker (none \| overlap \| model \| reverse \| random), top_k (passages given to the generator, default 3), generator (model \| extractive \| no_context) and model (a running server model). random, none, reverse and no_context are broken pieces, to check the attribution. {kind: faustus}: the document RAG of a running Faustus: url (default the setting faustus.url), source (documents \| brain), depth, answer (faustus: its own model answers; model: the generator given here). {kind: http}: any pipeline behind one request: url, method, headers, body (with {question} and {k}), passages_path, answer_path and optionally ranked_path, context_path, text_field, id_field, doc_field, score_field, top_k; give model to write the answer here when the pipeline only retrieves. |
| `question` (string) | yes |  |
| `set` (string) | no | The case set whose documents a builtin pipeline uses. |

## `rag_run_start`

Evaluate a RAG pipeline on a case set, piece by piece. Evaluar una tubería RAG.

Every accepted case is asked; per piece it measures retrieval (recall@k, MRR, nDCG), reranking (the same before and after) and generation (faithfulness to the passages, coverage of the expected answer, abstention), and attributes each failure to the piece that caused it. Runs in the background: follow it with rag_run_status and read rag_run_result.
Sinónimos: qué pieza falla, evaluar RAG de Faustus, evaluar por HTTP, recall, fidelidad, alucinación, atribución de fallos

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `set` (string) | yes |  |
| `pipeline` (object) | yes | The pipeline to evaluate. {kind: builtin}: this bench's own plain pipeline over the documents of the case set: retriever (bm25 \| random \| none), chunk_chars and chunk_overlap (fragment size), depth (passages retrieved, default 10), reranker (none \| overlap \| model \| reverse \| random), top_k (passages given to the generator, default 3), generator (model \| extractive \| no_context) and model (a running server model). random, none, reverse and no_context are broken pieces, to check the attribution. {kind: faustus}: the document RAG of a running Faustus: url (default the setting faustus.url), source (documents \| brain), depth, answer (faustus: its own model answers; model: the generator given here). {kind: http}: any pipeline behind one request: url, method, headers, body (with {question} and {k}), passages_path, answer_path and optionally ranked_path, context_path, text_field, id_field, doc_field, score_field, top_k; give model to write the answer here when the pipeline only retrieves. |
| `judge` (string/object) | no | The model that judges the answers (a running server). Empty: the setting judge.contestant; without one, word overlap. |
| `judge_mode` (auto \| logit \| model \| lexical) | no | logit: the logit judge (scores read from next-token probabilities; the server must return logprobs). model: a generated JSON verdict. lexical: word overlap, no model. auto: the first that works, in that order. |
| `ks` (array/null) | no | Cut-offs for recall@k and nDCG@k. Default 1, 3, 5, 10. |
| `match_threshold` (number/null) | no | Share of a reference passage's content words a retrieved passage must hold to count as it. Default 0.7. |
| `faithful_min` (number/null) | no | Faithfulness below this is an invented answer. Default 0.6. |
| `coverage_min` (number/null) | no | Coverage of the expected answer below this is an incomplete answer. Default 0.6. |
| `parallel` (integer/null) | no | Cases asked at the same time. Default 4. |
| `label` (string) | no |  |
| `include_proposed` (boolean) | no | Also run the cases nobody has reviewed yet. |
| `limit` (integer) | no | Only the first N cases (0: all). |
| `wait_s` (number) | no | Wait this long for the run to finish before answering; it keeps going afterwards. |

## `rag_run_status`

Progress of a RAG pipeline run. Estado de una evaluación RAG.

Sinónimos: cómo va, progreso de la evaluación RAG

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes | Run id (rr_…). |

## `rag_run_result`

A RAG run by piece: retrieval, reranking, generation and what each failure is due to. Resultado RAG.

summary.attribution says how many failures belong to the retriever (the passage never arrived), to the ranking (it arrived but was left outside the cut) and to the generator (it had the passage and abstained, invented or answered something else), with the share of each. With case, the whole trace of one case.
Sinónimos: el N % de los fallos son del recuperador, por qué falla, traza de un caso, pasajes recuperados

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes |  |
| `case` (string) | no | A case id: its whole trace (retrieved passages with the reference ones marked, order after the reranker, passages given to the generator, answer, scores). |
| `only_failed` (boolean) | no | List only the failed cases. |

## `rag_run_list`

Recent RAG pipeline runs with pass rate and main failing piece. Historial de evaluaciones RAG.

Sinónimos: evaluaciones RAG anteriores

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `set` (string) | no |  |
| `limit` (integer) | no |  |

## `rag_run_cancel`

Cancel a RAG pipeline run: finished cases are kept. Cancelar evaluación RAG.

Sinónimos: parar evaluación RAG

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes | Run id (rr_…). |

## `rag_run_delete`

Delete a RAG pipeline run and its results (confirm=true). Borrar evaluación RAG.

Sinónimos: quitar ejecución RAG

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes |  |
| `confirm` (boolean) | no |  |

## `rag_compare`

Compare two RAG runs case by case: fixed, broken, and in which piece. Comparar dos tuberías RAG.

For two configurations over the same case set (two fragment sizes, with and without reranker, two models): cases improved and regressed with an exact sign test, the change of every piece's measures, where the failures moved, and what differs between the two pipelines.
Sinónimos: comparar configuraciones, tamaño de fragmento, con y sin reordenador, antes y después

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `a` (string) | yes | The baseline run. |
| `b` (string) | yes | The run compared with it (another fragment size, with a reranker, another model...). |

## REST routes for the UI

- `GET /api/health`, `GET /api/status`
- `GET /api/dashboard` — the routing table as it stands, the run in progress, the queue, notices, what was never measured or went stale, the GPUs and the scheduler.
- `POST /api/dashboard/visit` — marks the notices seen.
- `GET /api/runs/{id}/events?after=N` — the run (state, progress per model) and the results written after result id `N`, without the answer texts; the run page polls it while the run is live.
- `GET /api/vision/{case}/{n}.png` — image `n` (from 1) of a case: a stored image or the one a vision case generates at run time; same-origin only.
- `POST /api/ui/call` `{name, arguments}` — any tool above, uncapped (the UI is local).
- `GET /manifest.webmanifest`, `GET /sw.js` — installable app.
