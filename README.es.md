<img src="app-icon.png" width="96" alt="">

# Pygmalion's Hoard

Estudio local para adaptar modelos de lenguaje (datasets, entrenamiento LoRA/QLoRA, fusión, cuantización, publicación) con un banco de evaluación incorporado. [English](README.md)

## Qué es

Pygmalion's Hoard prepara datasets, entrena adaptadores LoRA y QLoRA, fusiona adaptadores y modelos completos, amplía la ventana de contexto, convierte a GGUF, cuantiza con una matriz de importancia calculada sobre tus propios textos, mide el resultado frente a su origen y publica al ganador en Ollama o como servidor llama.cpp. Cada archivo que produce conserva su linaje y la receta que lo hizo se puede exportar.

Contiene también todo el banco de pruebas de modelos (el antiguo Galton's Hoard): descubrimiento de modelos, suites, importaciones de benchmarks, ejecuciones, estadística, arena, enrutado, usuarios simulados, matrices de arneses y campañas de agentes. Entrenamiento y Evaluación son funciones independientes en un solo proceso en el puerto 5202, más un interruptor de Evaluación automática para el descubrimiento periódico y el mantenimiento. Las tres están desactivadas la primera vez; arranca la app desde el Hoard Hub y activa lo que quieras. Abrir un módulo nunca reanuda trabajos antiguos en cola; reanúdalos explícitamente.

Funciona en tu PC, guarda sus datos en `data/`, va por sí sola en el navegador y se puede controlar con un asistente por MCP. Las páginas viven en `/training/` y `/evaluation/`.

## Qué hace

### Entrenamiento
- **Datasets.** Fuentes: archivos sueltos, una carpeta, JSONL o CSV, herramientas de otras apps Hoard a través del hub y ejemplos sintéticos escritos por un modelo profesor a partir de tus textos (quedan pendientes hasta que los apruebes). Operaciones ordenadas: eliminación de duplicados exactos y casi exactos, filtros de longitud e idioma, enmascarado o descarte de datos personales (correos, teléfonos, IBAN, tarjetas, DNI/NIE/CIF), con estadísticas, revisión, edición y versiones.
- **Entrenamiento.** LoRA y QLoRA (4 bits NF4) sobre modelos de lenguaje causales, con pérdida solo en los tokens del asistente en datos de chat, empaquetado para texto crudo, evaluación periódica, puntos de control reanudables, planificación coseno y semilla fija. `train_plan` estima la memoria de GPU y el tiempo y dice qué GPU permitida cabe.
- **Fusión.** Un adaptador en su base sin cargar el modelo entero, o varios modelos de la misma arquitectura con linear, SLERP, TIES o DARE, tras una comprobación de compatibilidad.
- **Ampliación de contexto.** Una copia de un modelo con escalado rope YaRN para 2x, 4x u 8x el contexto, una tabla de memoria de caché KV por longitud y resultados de aguja en un pajar.
- **GGUF y cuantización.** Conversión con el llama.cpp instalado, matriz de importancia en una GPU arrendada, varios tipos de cuantización a la vez (de Q8_0 a IQ3_M) y perplejidad con su error.
- **Evaluación de resultados.** Cada resultado GGUF o de Ollama se compara con la base de la que salió, escrita como el resultado, y el veredicto (mejor, peor, sin diferencia clara) con su intervalo se guarda en el resultado. Los archivos de referencia que faltan se construyen dentro de la misma tubería.
- **Captura.** Un proxy de chat-completions delante de un servidor llama.cpp o vLLM que convierte turnos reales de Faustus, de los Hoards o de cualquier arnés de agentes en datos de entrenamiento con máscara de pérdida, ids de tokens y log-probabilidades, un árbol de llamadas por sesión, recompensas y exportaciones filtradas.
- **Publicación.** Un Modelfile y `ollama create` con la etiqueta `pyg-<nombre>:<etiqueta>`, o una entrada de servidor llama.cpp en la lista de backends del hub (nunca se arranca sola). Despublicar pide confirmación.
- **Trabajos.** Dos carriles, `gpu` y `cpu`. Las tuberías encadenan entrenar, fusionar, convertir, matriz, cuantizar, perplejidad, publicar y evaluar; un fallo detiene el resto y la tubería se puede reanudar. Cancelar un entrenamiento guarda un punto de control; los trabajos que se encuentran en marcha tras un reinicio se marcan como interrumpidos.
- **GPU.** Solo se usan las GPU permitidas, siempre mediante un arrendamiento del hub de la familia; las GPU reservadas al dueño del ordenador necesitan confirmación explícita.
- **Linaje.** Base, adaptador, fusionado, variante de contexto, GGUF, matriz y artefactos de Ollama forman un grafo con el hash del dataset, los parámetros y el resumen del entrenamiento.

### Evaluación
- Descubrimiento de modelos (servidores, Ollama, archivos GGUF), suites e importaciones de benchmarks, ejecuciones con reanudación, resultados, estadística y comparación, y exportación de resultados a CSV o JSON.
- Arena, enrutado, prueba de reglas, usuarios simulados, matrices de arneses (también en un eje de idioma español/inglés), un banco de vídeos hecho por código y campañas de agentes.
- Comprobaciones de respuesta fuera de una ejecución: fidelidad (afirmación a afirmación frente al contexto), relevancia de la respuesta y similitud con una referencia (ROUGE-L, solapamiento de palabras o razón de caracteres).
- Evaluación de tuberías RAG: mide el recuperador (recall@k, MRR, nDCG), el orden y el corte, y el generador (fidelidad, cobertura, abstención), y atribuye cada caso fallido a la pieza que lo perdió. Evalúa una tubería incorporada, el RAG de documentos de un Faustus en marcha o cualquier tubería detrás de una petición HTTP.
- Evaluación de trajes para Darwin's Hoard: casos con comprobaciones de aceptación ejecutables, tres brazos (herramientas genéricas, traje actual, candidato), agregados con su dispersión y resultados enviados a Darwin's Hoard.
- Las evaluaciones de entrenamiento llaman al evaluador directamente en el proceso. Las herramientas de evaluación se listan abajo y en [docs/evaluation/API.md](docs/evaluation/API.md).

### Pantallas
Español por defecto, inglés con un clic, modo oscuro, escritorio y móvil. Páginas: Panel, Modelos base (con búsqueda y descarga en Hugging Face), Datasets, Entrenar, Trabajos, Fusionar, Cuantizar, Contexto, Linaje, Captura y Ajustes, más las páginas de Evaluación.

### Límites
- No instala PyTorch ni llama.cpp. Los trabajadores de entrenamiento, fusión y sondeo corren en un entorno Python aparte que preparas tú; la app dice qué falta y el comando para arreglarlo.
- Solo entrena modelos de lenguaje decodificadores desde puntos de control safetensors que el `transformers` instalado pueda cargar; los pesos basados en pickle se rechazan.
- YaRN cambia la configuración, no los pesos; la calidad en el extremo de la ventana hay que medirla con el módulo de Evaluación.
- La evaluación con datos apartados necesita un modelo juez configurado; sin él, la ejecución se rechaza indicando el ajuste que hay que cambiar.
- La evaluación, las fuentes de apps de la familia y los datos sintéticos necesitan el hub o la otra app en marcha; sin ellos esos pasos fallan con un mensaje claro y el resto sigue funcionando.
- La estimación de tiempo es una constante de rendimiento aproximada que las mediciones sustituyen mientras corre el trabajo.
- La captura necesita un servidor llama.cpp (`llama-server`) o vLLM al que pueda llegar; en llama.cpp las herramientas solo se renderizan con un servidor arrancado con `--jinja`, y las llamadas que no puede verificar se marcan como no verificadas.

## Cómo ejecutarlo

Requiere Python 3.11 o posterior; Node 22 solo para recompilar la interfaz.

```bash
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt      # Windows; venv/bin/python en otros
venv/Scripts/python -m pygmalion_hoard                        # http://127.0.0.1:5202
python scripts/launch.py                                      # puerto libre y abre el navegador
```

El entorno del entrenador es aparte. Créalo una vez y apunta `env.python` en Ajustes a él:

```bash
python -m venv <trabajo>/venv
<trabajo>/venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
<trabajo>/venv/Scripts/python -m pip install -U transformers peft trl accelerate datasets bitsandbytes safetensors huggingface_hub gguf numpy
```

Para trabajar con GGUF fija `llama.bin_dir` (llama-quantize, llama-imatrix, llama-perplexity, llama-server) y `llama.src_dir` (fuentes de llama.cpp con los scripts de conversión). Ajustes y `env_check` muestran exactamente qué falta.

La interfaz viene compilada en `pygmalion_hoard/static`; recompílala con `npm install && npx vite build`. Un segundo arranque con otro en marcha no toca nada: dice dónde está la copia en marcha y sale.

Entorno: `PYGMALION_PORT` (5202), `PYGMALION_DATA_DIR`, `PORT_STRICT=1`, `PYGMALION_SCHEDULER=0` (sin trabajo en segundo plano), `PYGMALION_OFFLINE=1` (sin red), `PYGMALION_WORKERS_DIR`, `PYGMALION_ALLOWED_HOSTS`, `PYGMALION_HTTP_TIMEOUT_S`, `PYGMALION_HF_TOKEN` (o guárdalo en Ajustes). Todo lo demás es un ajuste guardado en `data/pygmalion.db`.

Todo se queda en el ordenador. `data/` contiene la base de datos, las versiones de datasets, los registros, el token MCP y `secrets.env` (el token de Hugging Face, con permisos solo del propietario, que nunca se muestra ni se registra). Las carpetas de datos existentes de entrenamiento y evaluación se pueden adoptar en su sitio mediante `data/modules.json` (`python scripts/adopt_evaluation.py --evaluation-data <carpeta>`), sin copia ni conversión. Véase [docs/FUSION.md](docs/FUSION.md).

## Conexión con Faustus y el Hoard Hub

- `faustus-plugin.json` es el manifiesto. `mcp_server.py` es un puente MCP por stdio llamado `pygmalion-hoard` que expone el catálogo estático combinado; nunca abre la base de datos ni arranca el servidor: reenvía cada llamada a la app en marcha con el token de `data/mcp-token`. Las herramientas que inician un trabajo aceptan `wait_s`, como máximo 150 segundos.
- Borrar requiere `confirm=true`. Permitir una GPU del propietario requiere `confirm_reserved=true`, que un asistente no debe pasar salvo que el propietario lo haya dicho.
- Un botón «Faustus» abre el panel de asistente común de la familia, con el modelo base, dataset, trabajo o artefacto en pantalla y los campos de entrenamiento sin guardar; las listas se refrescan tras cada turno.
- Los arrendamientos de GPU y el servicio de modelo global configurado vienen del hub. La app lleva skills para agentes conectados (`model-finetune-and-eval`, `eval-harness`, `prompt-optimizer`, `measured-improvement`, `cross-model-review`).

Herramientas de entrenamiento (50): `pygmalion_overview`, `env_check`, `bases_list`, `base_get`, `hf_search`, `base_download`, `datasets_list`, `dataset_get`, `dataset_create`, `dataset_records`, `dataset_review`, `dataset_preview_source`, `dataset_apply`, `dataset_delete`, `train_plan`, `train_start`, `merge_check`, `merge_lora_start`, `merge_models_start`, `ctx_extend_start`, `ctx_fit`, `convert_start`, `quantize_start`, `perplexity_start`, `jobs_list`, `job_get`, `job_cancel`, `job_resume`, `job_delete`, `artifacts_list`, `artifact_get`, `artifact_recipe`, `artifact_update`, `artifact_delete`, `lineage_graph`, `evaluate_plan`, `evaluate_start`, `publish_ollama`, `publish_llama`, `unpublish`, `settings_get`, `settings_set`, `secret_set`, `gpu_status`, `capture_proxy`, `capture_sessions`, `capture_session`, `capture_outcome`, `capture_export`, `capture_delete`. Argumentos en [docs/API.md](docs/API.md).

Controles de módulo: `pygmalion_modules`, `pygmalion_set_module`. Donde los nombres chocan con herramientas de evaluación, las de evaluación son `evaluation_settings_get`, `evaluation_settings_set` y `evaluation_gpu_status`.

<details>
<summary>Herramientas de evaluación</summary>

`galton_overview`, `galton_status`, `models_list`, `models_refresh`, `model_get`, `model_add`, `model_update`, `model_remove`, `suites_list`, `suite_get`, `case_get`, `suite_create`, `suite_update`, `suite_duplicate`, `suite_remove`, `case_add`, `case_update`, `case_remove`, `cases_import`, `benchmark_import`, `case_try`, `checkers_list`, `answer_check`, `run_plan`, `run_start`, `run_status`, `run_cancel`, `run_discard`, `run_restore`, `run_resume`, `runs_list`, `run_results`, `run_results_export`, `galton_run`, `measure_new`, `judge_run`, `leaderboard`, `compare`, `gates_correlation`, `recommend`, `routes_get`, `routes_publish`, `arena_next`, `arena_vote`, `arena_ratings`, `settings_get`, `settings_set`, `gpu_status`, `evaluation_settings_get`, `evaluation_settings_set`, `evaluation_gpu_status`, `notices_list`, `housekeeping_run`, `rulebench_tasks`, `rulebench_score`, `harness_catalog`, `harness_matrix_start`, `harness_matrix_status`, `harness_matrix_result`, `harness_cell`, `harness_matrix_cancel`, `harness_matrix_list`, `langbench_task`, `langbench_judge`, `videobench_catalog`, `videobench_prompt`, `videobench_start`, `videobench_status`, `videobench_result`, `videobench_item`, `videobench_cancel`, `videobench_list`, `logit_judge_run`, `logit_judge_result`, `logit_judge_list`, `logit_judge_probe`, `logit_judge_cancel`, `logit_judge_delete`, `agentbench_catalog`, `agentbench_start`, `agentbench_status`, `agentbench_result`, `agentbench_cell`, `agentbench_trial`, `agentbench_sync`, `agentbench_cancel`, `agentbench_list`, `pygmalion_suit_eval_define`, `pygmalion_suit_library`, `pygmalion_suit_eval_list`, `pygmalion_suit_eval_get`, `pygmalion_suit_eval_record`, `pygmalion_suit_eval_check`, `pygmalion_suit_eval_score`, `pygmalion_suit_eval_export`, `pygmalion_suit_eval_delete`, `rag_sets`, `rag_set_save`, `rag_set_get`, `rag_set_delete`, `rag_cases_generate`, `rag_case_save`, `rag_cases_review`, `rag_case_remove`, `rag_probe`, `rag_run_start`, `rag_run_status`, `rag_run_result`, `rag_run_list`, `rag_run_cancel`, `rag_run_delete`, `rag_compare`

</details>

## Estructura del proyecto

| Ruta | Contenido |
| --- | --- |
| `pygmalion_hoard/` | Paquete Python: API, entrenamiento, trabajos, linaje, captura, interfaz estática |
| `galton_hoard/` | Módulo de evaluación |
| `client/`, `evaluation-client/` | Interfaces de entrenamiento y evaluación |
| `mcp_server.py` | Puente MCP por stdio |
| `faustus-plugin.json` | Manifiesto del plugin de Faustus |
| `compatibility/` | Manifiestos de compatibilidad de antiguas identidades de app |
| `scripts/` | Lanzador, generador de la documentación de la API, ayudante de adopción |
| `docs/` | Referencias de arquitectura, fusión y API |
| `provenance/` | Inventario de orígenes |
| `tests/`, `evaluation_tests/`, `fusion_tests/` | Pruebas |

## Desarrollo

```bash
python -m pytest -q                  # toda la batería; sin red, sin GPU, sin torch real
python scripts/gen_api_doc.py        # regenera docs/API.md tras cambiar una herramienta
npx vite build                       # recompila la interfaz
```

FastAPI, SQLite (WAL) y un gestor de trabajos de dos carriles; React 19, Vite y Tailwind para la interfaz. Véanse [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) y [AGENTS.md](AGENTS.md). Licencia: MIT.
