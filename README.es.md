# Pygmalion's Hoard

## Estudio unificado: Pygmalion y Galton

Pygmalion incorpora todas las funciones de Galton de forma nativa. Un servidor en el puerto 5202, sin ejecutar el servidor anterior. En la pantalla **Funciones** hay tres controles independientes: Entrenamiento, Evaluación y Evaluaciones automáticas. Todos están desactivados en la primera apertura; el catálogo MCP no inicia el servidor ni sus módulos. Activa sólo lo que quieras usar desde el Hub y desde esta pantalla.

Entrenamiento conserva datasets, LoRA/QLoRA, fusiones, contexto, conversión, cuantización, captura, linaje y publicación. Evaluación conserva modelos y suites, importaciones, ejecuciones y reanudación, resultados, comparación y estadísticas, arena, rutas, pruebas de reglas, usuarios simulados, matrices de arneses y campañas de agentes. Las evaluaciones de entrenamiento llaman a este motor dentro del mismo proceso. La vigilancia periódica necesita además el control de Evaluaciones automáticas; los ajustes históricos permanecen intactos y no reanudamos colas antiguas al abrir un módulo. Los trabajos anteriores se pueden reanudar explícitamente.

Las páginas originales se encuentran en `/training/` y `/evaluation/`, sin marcos embebidos. Se conservan `/capture/...` para los arneses existentes. El catálogo combinado expone **117 herramientas**: las 50 de entrenamiento, 65 de evaluación y `pygmalion_modules` / `pygmalion_set_module`. Tres colisiones se publican con prefijo: `evaluation_settings_get`, `evaluation_settings_set`, `evaluation_gpu_status`.

Los datos se utilizan en su ubicación original mediante `data/modules.json` (`data_paths.training` y `data_paths.evaluation`). No se copian ni convierten bases de datos, datasets, modelos, resultados o linaje. `python scripts/adopt_evaluation.py --evaluation-data <carpeta>` configura la ubicación con todos los módulos OFF. Desactivar detiene workers y solicitudes antes de cerrar sus bases; los datos permanecen.

Esta fusión no cambia las leases de GPU ni el servicio global de modelos. Las pruebas del producto usan datos inventados y motores simulados; no certifican nuevos resultados de un modelo real. Inventario y arquitectura: [fusión](docs/FUSION.md), [API de evaluación](docs/evaluation/API.md), [procedencia](provenance/galton/source.json).

### Herramientas de evaluación (65):

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

Un estudio local para adaptar modelos de lenguaje. Prepara datasets, entrena adaptadores LoRA y QLoRA, fusiona adaptadores y modelos enteros, amplía la ventana de contexto, convierte a GGUF, cuantiza con una matriz de importancia calculada sobre tus propios textos, mide el resultado frente a su origen y publica el mejor en Ollama o como servidor llama.cpp. Cada archivo que produce conserva su linaje, y la receta que lo creó se puede exportar y repetir.

Forma parte de la familia de apps locales Hoard: funciona en tu ordenador, guarda sus datos en `data/`, se usa sola desde el navegador y puede manejarla un asistente por MCP.

[English version](README.md)

## Qué hace

- **Datasets.** Fuentes: archivos sueltos, una carpeta, JSONL y CSV pegados o en archivo, herramientas de otras apps Hoard (a través del hub de la familia) y ejemplos sintéticos que escribe un modelo profesor a partir de tus textos. Las operaciones se aplican en orden: quitar duplicados exactos y casi duplicados (MinHash sobre n-gramas de 5), filtros de longitud e idioma, y enmascarar o descartar datos personales (correos, teléfonos, IBAN, tarjetas, DNI/NIE/CIF; los identificadores solo cuentan cuando su dígito de control es correcto, y un teléfono cuando lleva prefijo de país, una etiqueta o la agrupación de cifras habitual). El reparto entre entrenamiento y evaluación usa semilla. Las versiones son inmutables; revisar registros (aceptar, rechazar, editar) crea una versión nueva. Estadísticas: registros, tokens, histograma de longitudes, reparto de roles e idiomas.
- **Entrenamiento.** LoRA y QLoRA (4 bits NF4, cálculo en bfloat16) sobre un modelo de lenguaje causal, con pérdida solo sobre los turnos del asistente en datos de chat, empaquetado en texto plano, evaluación periódica, puntos de control para reanudar, calendario coseno y semilla fija. Antes de empezar, `train_plan` estima la memoria de GPU con su fórmula, estima el tiempo e indica qué GPU permitida cabe. El tiempo sale de los tokens que hay que procesar (los registros que recorren los pasos x el registro medio, cortado en `seq_len`, x relleno) divididos por un modelo de velocidad: 1 600 tokens/s por cada mil millones de parámetros en QLoRA sobre una tarjeta de 16 GB de la serie 50, 1,6 veces más con LoRA normal, escalado por la velocidad de la tarjeta y acotado para modelos muy pequeños, más un arranque y un coste por paso. Las constantes son aproximadas y el plan muestra la fórmula con sus números. Cuando el trabajo arranca, los tokens/s medidos y el tiempo restante real sustituyen a la estimación (la vista del trabajo dice cuál muestra), y un trabajo terminado muestra cuánto duró frente a lo estimado.
- **Fusiones.** Un adaptador en su base (tensor a tensor, sin cargar el modelo entero), o varios modelos de la misma arquitectura con lineal, SLERP, TIES o DARE. Antes se comprueba la compatibilidad.
- **Ampliación de contexto.** Una copia del modelo con YaRN para 2, 4 u 8 veces el contexto, una tabla de memoria de la caché KV por longitud a partir de los metadatos del GGUF y los resultados de la aguja en un pajar por longitud.
- **GGUF y cuantización.** Conversión con el llama.cpp instalado (antes se comprueba la arquitectura), matriz de importancia en una GPU reservada, varios tipos de cuantización de una vez (Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, Q3_K_M, IQ3_M) y perplejidad con su error. La perplejidad se guarda en el archivo medido con su error, texto, contexto y fragmentos; cuando una cadena cuantiza a varios tipos con perplejidad activada, se mide igual el archivo sin cuantizar y cada cuantización, y el archivo propio del modelo base cuando existe o se construye para la evaluación. Linaje y Cuantizar los muestran juntos, con el cambio respecto al archivo padre (lo que costó la cuantización) y respecto al del base del mismo tipo (lo que hizo el ajuste).
- **Evaluación.** Cada resultado GGUF o de Ollama se compara a través de Galton's Hoard (por el hub, o directamente si no está), y el veredicto (mejor, peor, sin diferencia clara) con su diferencia e intervalo se guarda en el resultado junto con el tipo de referencia en que se apoya. Con qué se compara depende de lo que se pregunte. Las intenciones que preguntan si un entrenamiento ayudó o perjudicó (`dataset`, `style`, `writing`, `code`, `general`, `smoke`) comparan con el **modelo base del que partió el entrenamiento, escrito como el resultado**: el mismo tipo de cuantización y, si el resultado usó una matriz de importancia, una matriz calculada con el mismo texto de calibración. El antecesor GGUF más cercano de un archivo ajustado es el f16 del mismo ajuste, que mediría la cuantización y no el entrenamiento. Si el base aún no tiene un archivo así, `evaluate_start` encola los pasos que lo hacen (convertir la carpeta del base, matriz, cuantizar; lo que ya existe se reutiliza) como una sola cadena que termina en la evaluación; los archivos quedan en el linaje bajo el modelo base y la siguiente evaluación los encuentra. `context` compara con el modelo anterior a la extensión de contexto, construido igual. Una cuantización sola (sin entrenamiento detrás; intención `quant`, su valor por defecto) sigue comparándose con el f16 del que salió. `against` manda siempre. `evaluate_plan` (y `artifact_get`, como `evaluate_plan`) dice qué referencia usaría una evaluación y si hay que prepararla antes; el cuadro «Evaluar» de Linaje muestra esa frase. Un resultado cuyo modelo base no está en el linaje se compara con su archivo padre y el veredicto lleva un aviso. Un resultado que desciende de un entrenamiento cuyo dataset reserva registros se mide por defecto con esos registros (`intent: dataset`): los registros reservados de esa versión del dataset se convierten en una suite de Galton `pyg-<dataset>-v<n>-eval` (se crea una vez por versión y se reutiliza; cada registro es un caso con la respuesta de referencia y un comprobador de juez), el padre y el hijo la responden y una suite general pequeña (`rapida`) corre a su lado para detectar regresiones. Se guardan y se muestran los dos veredictos; un resultado general peor detiene la sugerencia de promover. El juez es el modelo propio de Galton (su ajuste `judge.contestant`): sin uno, la evaluación se detiene antes de empezar y dice dónde configurarlo. `galton.eval_cases` limita el número de casos. Sobre los registros reservados los dos modelos responden sin razonar antes (`effort: off`, como están escritos los registros) salvo que la evaluación pase sus propios ajustes.
- **Captura.** Un proxy de chat completions delante de un servidor llama.cpp o vLLM, para convertir los turnos reales de Faustus, de los Hoards o de cualquier arnés de agentes en datos de entrenamiento con máscara de pérdida. Se apunta el arnés a `http://127.0.0.1:5202/training/capture/v1` (una sola sesión), `/capture/<sesión>/v1` o `/capture/<arnés>/<sesión>/v1`; la sesión y el arnés también pueden ir en la clave de API (`sesión` o `arnés:sesión`) o en las cabeceras `X-Capture-Session` y `X-Capture-Harness`. Cada llamada se reenvía al servidor configurado sin streaming (un cliente que pidió streaming recibe la respuesta completa re-emitida como eventos, también las llamadas a herramientas) y el cliente recibe la respuesta normal, sin los campos extra; sus parámetros de muestreo no se tocan salvo que `capture.faithful_sampling` esté activo, que anula `top_k`, `top_p` y `min_p` para que las log-probabilidades guardadas describan la distribución de la que salió cada token (para aprendizaje por refuerzo). Cada llamada se guarda como un nodo: los ids del prompt (tras la plantilla de chat), los ids y log-probabilidades de lo que muestreó el modelo, los mensajes, las herramientas, los parámetros de muestreo, el modelo, el servidor y un estado de verificación. vLLM da los ids directamente (`prompt_token_ids`, `token_ids`). llama.cpp da los ids y las log-probabilidades de los tokens muestreados pero no los del prompt: se reconstruyen con `/apply-template` y `/tokenize` del propio servidor y la cuenta se comprueba con `usage.prompt_tokens`; una llamada que no coincide (herramientas en un servidor arrancado sin `--jinja`, imágenes, una plantilla poco habitual) se guarda como *sin verificar* y se deja fuera de las exportaciones salvo que se pida. Cuando llama.cpp cuenta un token generado más de los que lista, es el de fin de turno: se añade (su log-probabilidad es desconocida, así que va a nulo) para que el modelo aprenda a parar. Una llamada cuelga de otra anterior de su sesión cuando todos los tokens de aquella (prompt y respuesta) son el comienzo exacto de su prompt; las llamadas de una sesión forman un árbol de prefijos, las hermanas son ramas (reintentos, alternativas, subagentes) y una llamada que no continúa ninguna empieza una raíz nueva. Cada camino de raíz a hoja es una secuencia de entrenamiento: el contexto lleva máscara 0, los tokens que muestreó el modelo máscara 1, y un token compartido por varios caminos se entrena una vez. Las sesiones admiten una `reward` y un `outcome` opcionales (se fijan después con `capture_outcome`; vacíos cuando nada verificó el resultado). La exportación (`capture_export`; `dry_run` informa antes del reparto) filtra por sesión, arnés, fecha y recompensa, puede equilibrar por arnés (`balance_by_harness`: tantas secuencias por arnés como tenga el menor, o un tope, con muestra de semilla fija e informe por arnés) y escribe en `data/capture/exports/` un fichero de tokens para SFT (`input_ids`, `labels` con -100 en el contexto), un fichero de tokens para RL (máscara de pérdida, log-probabilidades por token, recompensa) y, si se da nombre de dataset, un dataset de chat registrado en Datasets con el que Entrenar puede entrenar. Las llamadas se guardan como la parte que sigue a su padre, así que una sesión larga de agente cuesta sus tokens nuevos y no todo su contexto en cada paso.
- **Publicación.** Un Modelfile y `ollama create` con la etiqueta `pyg-<nombre>:<etiqueta>`, o una entrada de servidor llama.cpp en la lista de backends del hub (nunca se arranca sola, solo en GPU permitidas). Retirar una publicación pide confirmación.
- **Trabajos.** Dos carriles, un trabajo en cada uno: `gpu` y `cpu`. Una cadena encadena pasos (entrenar, fusionar, convertir, matriz, cuantizar, perplejidad, publicar, evaluar); si uno falla, los siguientes no empiezan y se puede reanudar. Cancelar un entrenamiento guarda antes un punto de control. Los trabajos que estaban en marcha al reiniciar quedan como interrumpidos y se pueden reanudar. La página de un trabajo lista todos los pasos de su cadena, también los que aún no se han creado, con su estado actual, y se refresca mientras la cadena corre. El planificador lee lo que corre de la base de datos, cierra las filas que dicen correr sin que nada las ejecute y la insignia de Trabajos cuenta solo los que corren o esperan. La salida de los programas se limpia antes de guardarla: se quitan las secuencias de escape del terminal y las líneas de giro o progreso que un programa redibuja en el sitio (por ejemplo `ollama create`) se quedan en su último estado. Cada trabajo informa al hub de la familia con los eventos canónicos `pygmalion.job.queued|started|progress|done|failed|cancelled` (`{job_id, title, kind, progress, gpu, eta_s, url, error}`; `kind` es `train`, `merge`, `convert`, `quantize` o `publish`, y un publish terminado lleva `model`, el nombre con el que se publicó), que muestra la pestaña Trabajos del hub y que una regla del hub usa para pedir a Galton que mida un modelo recién publicado.
- **GPU.** Solo se usan las GPU permitidas (por defecto la 2 y la 3), siempre con una reserva del hub de la familia. Las GPU 0 y 1 son del dueño del ordenador y piden confirmación expresa para permitirlas.
- **Linaje.** Los artefactos base, adaptador, fusionado, variante de contexto, GGUF, matriz y Ollama forman un grafo con la huella del dataset, los parámetros y el resumen del entrenamiento. La receta exportada lista todos los pasos desde la base, en orden.

## Pantallas

En castellano por defecto, en inglés con un clic, oscuro, en escritorio y en móvil.

- **Panel**: estado del entorno, GPU con sus reservas, trabajos en marcha con una miniatura de la pérdida, últimos resultados con su veredicto, accesos rápidos.
- **Modelos base**: modelos locales con arquitectura, tamaño y lo que admiten; búsqueda y descarga en Hugging Face con confirmación del tamaño.
- **Datasets**: lista, constructor con fuentes y operaciones, estadísticas, registros con revisión y edición, versiones.
- **Entrenar**: base, versión del dataset, método y parámetros con el plan y la estimación de memoria, pasos posteriores, inicio.
- **Trabajos**: cola e historial; cada trabajo con sus curvas de pérdida, progreso, tiempo restante, memoria de GPU, registro, cancelar y reanudar.
- **Fusionar**: adaptador en la base; mezcla de modelos con la comprobación de compatibilidad.
- **Cuantizar**: elegir tipos, matriz y texto de calibración; tabla de resultados con tamaño, perplejidad y veredicto.
- **Contexto**: crear una variante de contexto, ver la tabla de memoria y los resultados de la aguja por longitud.
- **Linaje**: grafo y tabla de artefactos, detalle con exportación de la receta, evaluación, publicación y notas.
- **Captura**: el proxy de captura (servidor que se captura, iniciar y parar, las direcciones para pegar en un arnés con botón de copiar), sesiones con sus llamadas, tokens, sin verificar y recompensa, el árbol de llamadas de una sesión (ramas, raíces nuevas, estado de verificación; una llamada se abre en un panel con sus mensajes, herramientas, respuesta y los tokens muestreados coloreados por log-probabilidad), el formulario de recompensa y la exportación con filtros, equilibrio por arnés, una simulación que informa del reparto y los ficheros que escribió.
- **Ajustes**: GPU permitidas, rutas, valores por defecto, modelo profesor, enlace con Galton, token de Hugging Face (solo escritura), comprobación del entorno con el comando que arregla cada falta.

## Qué no hace

- No instala PyTorch ni llama.cpp. Los procesos de entrenamiento, fusión y sondeo se ejecutan en un entorno de Python aparte que preparas tú (ver Instalación); la app indica qué falta y el comando para arreglarlo.
- Solo entrena modelos de lenguaje decodificadores que el `transformers` instalado pueda cargar como modelo causal, desde checkpoints safetensors. Se rechazan los adaptadores y pesos basados en pickle.
- El entrenamiento, la vía de transformers de la fusión de adaptadores y todos los pasos con GPU se han probado con procesos simulados y programas de llama.cpp simulados, no con una GPU real. La primera ejecución real en tu equipo es la prueba de verdad.
- YaRN cambia la configuración, no los pesos: la calidad al final de la ventana hay que medirla, y para eso hace falta Galton's Hoard en marcha.
- Al publicar, `num_ctx` sale del ajuste `publish.num_ctx` (8192 por defecto, nunca por encima del contexto con el que se entrenó el modelo; editable en el formulario de publicar), porque Ollama reserva memoria para toda la ventana. Una variante de contexto se publica con la longitud para la que se construyó.
- La evaluación sobre los registros reservados necesita un modelo juez configurado en Galton; sin él se rechaza con el ajuste que hay que cambiar.
- Preparar el archivo de referencia del modelo base convierte y cuantiza ese modelo, lo que cuesta disco y tiempo; se hace una vez por modelo base, cuantización y texto de calibración. Una fusión de modelos enteros (sin entrenamiento) no tiene base con la que medirse y se compara con su archivo padre.
- La evaluación, las fuentes de otras apps y los datos sintéticos necesitan el hub de la familia o la otra app; sin ellos esos pasos fallan con un mensaje claro y el resto sigue funcionando.
- Los ejemplos sintéticos valen lo que valga el modelo profesor; quedan pendientes hasta que los apruebes.
- La estimación de tiempo es una velocidad de referencia aproximada (1 600 tokens/s por cada mil millones de parámetros en QLoRA sobre una tarjeta de 16 GB de la serie 50), no una medición de tu equipo; los primeros pasos del trabajo la sustituyen. El texto que llega de una biblioteca (un error de Python, un mensaje de Hugging Face, los avisos propios de Galton) se muestra tal como llegó, en su idioma.

- La captura necesita un servidor llama.cpp (`llama-server`) o vLLM al que pueda llegar; guarda los ids que el servidor informa y no tokeniza nada por su cuenta. En llama.cpp los ids del prompt se reconstruyen, no se informan: las herramientas solo las renderiza un servidor arrancado con `--jinja`, y un prompt con imágenes no coincide con su cuenta, así que esas llamadas quedan sin verificar. El enlace entre llamadas es un prefijo exacto de tokens: un arnés que reescribe un turno anterior (le quita el razonamiento, recorta un resultado de herramienta) cambia esos tokens y la cadena se rompe ahí, lo que empieza una raíz nueva. El dataset de chat que registra aplana las herramientas a texto (un resultado de herramienta pasa a ser un turno de usuario, una llamada a herramienta un bloque `<tool_call>`) porque el entrenador integrado lee turnos sencillos de sistema, usuario y asistente; los ficheros de tokens guardan los tokens exactos y sirven para un entrenador externo o para aprendizaje por refuerzo, el entrenador integrado no los lee. Los ids de token pertenecen al tokenizador del modelo del servidor: entrena con ellos solo un modelo con el mismo tokenizador. El proxy no tiene clave propia: contesta a quien llegue al puerto de la app (solo el propio equipo salvo que `PYGMALION_ALLOWED_HOSTS` diga otra cosa) y guarda cada prompt que reenvía, por eso está apagado hasta que se inicia y las sesiones se pueden borrar.

## Panel de Faustus

Un botón «Faustus» abre el panel de asistente común de la familia sin salir de la app. Faustus guarda las conversaciones
(también en Faustus Studio), ejecuta el agente con el modelo elegido globalmente y usa las herramientas de esta app. Cada
turno lleva el modelo base, dataset (y la versión en pantalla), trabajo o artefacto abierto; lo que esté sin guardar en la preparación del entrenamiento, el constructor de datasets y el nombre y las notas del artefacto. Al terminar un turno, las listas y la vista abierta se refrescan sin perder lo que estuvieras escribiendo.
El panel lo sirve Hoard Link (`install_spa`) y habla con Faustus a través del relay de la app, así que ningún token llega
al navegador.

## Instalación

Requisitos: Python 3.11 o superior (probado con 3.11 y 3.13), Node 22 solo para reconstruir la interfaz.

```bash
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt      # Windows; venv/bin/python en otros sistemas
```

El entorno de entrenamiento es aparte y este archivo no lo instala. Créalo una vez y apunta `env.python` en Ajustes a él:

```bash
python -m venv <trabajo>/venv
<trabajo>/venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
<trabajo>/venv/Scripts/python -m pip install -U transformers peft trl accelerate datasets bitsandbytes safetensors huggingface_hub gguf numpy
```

Para trabajar con GGUF indica `llama.bin_dir` (llama-quantize, llama-imatrix, llama-perplexity, llama-server) y `llama.src_dir` (fuentes de llama.cpp con `convert_hf_to_gguf.py` y `convert_lora_to_gguf.py`). Ajustes y `env_check` muestran exactamente qué falta; `env_check` lista todos los dispositivos CUDA que ve el entorno de entrenamiento (índice en el orden de nvidia-smi, nombre, memoria total). La comprobación de arquitecturas lee `convert_hf_to_gguf.py` y, en las fuentes más nuevas de llama.cpp, todo el paquete `conversion/`. Los contadores del panel leen de la misma fuente que `bases_list` y la página se recarga tras una comprobación del entorno y tras cualquier escritura que cambie un contador.

## Ejecución

```bash
venv/Scripts/python -m pygmalion_hoard                        # http://127.0.0.1:5202
python scripts/launch.py                                      # puerto libre y abre el navegador
```

La interfaz viene compilada en `pygmalion_hoard/static`. Para reconstruirla: `npm install && npx vite build`. Un segundo arranque de la app (o de `scripts/launch.py`) con otra en marcha no toca nada: avisa de dónde está la que corre y termina. El arranque, el guardián de peticiones, la ruta de salud, el formato de errores, los archivos de app instalable y el servidor de la página única son los compartidos de Hoard Link.

Entorno: `PYGMALION_PORT` (5202), `PYGMALION_DATA_DIR`, `PORT_STRICT=1`, `PYGMALION_SCHEDULER=0` (sin trabajo en segundo plano), `PYGMALION_OFFLINE=1` (sin red), `PYGMALION_WORKERS_DIR`, `PYGMALION_ALLOWED_HOSTS`, `PYGMALION_HTTP_TIMEOUT_S`, `PYGMALION_HF_TOKEN` (o guárdalo en Ajustes). Todo lo demás son ajustes guardados en `data/pygmalion.db`.

## Asistentes (MCP)

`mcp_server.py` es un puente MCP por stdio llamado `pygmalion-hoard` (el puente de catálogo compartido de Hoard Link). Nunca abre la base de datos: reenvía cada llamada a la app en marcha con el token de `data/mcp-token` (el token se crea una vez y se conserva entre arranques), y arranca la app cuando no responde. Las herramientas que lanzan un trabajo aceptan `wait_s`, como mucho 150 segundos: si el trabajo sigue en marcha al acabar la espera, la respuesta dice `still_running` y `job_get` lo sigue. `faustus-plugin.json` describe la app, su comprobación de salud y el puente para Faustus y el Hoard Hub.

Herramientas (50): `pygmalion_overview`, `env_check`, `bases_list`, `base_get`, `hf_search`, `base_download`, `datasets_list`, `dataset_get`, `dataset_create`, `dataset_records`, `dataset_review`, `dataset_preview_source`, `dataset_apply`, `dataset_delete`, `train_plan`, `train_start`, `merge_check`, `merge_lora_start`, `merge_models_start`, `ctx_extend_start`, `ctx_fit`, `convert_start`, `quantize_start`, `perplexity_start`, `jobs_list`, `job_get`, `job_cancel`, `job_resume`, `job_delete`, `artifacts_list`, `artifact_get`, `artifact_recipe`, `artifact_update`, `artifact_delete`, `lineage_graph`, `evaluate_plan`, `evaluate_start`, `publish_ollama`, `publish_llama`, `unpublish`, `settings_get`, `settings_set`, `secret_set`, `gpu_status`, `capture_proxy`, `capture_sessions`, `capture_session`, `capture_outcome`, `capture_export`, `capture_delete`. Los argumentos están en [docs/API.md](docs/API.md).

Borrar exige `confirm=true`. Permitir una GPU del dueño exige `confirm_reserved=true`, que un asistente no debe pasar salvo que el dueño lo haya dicho.

## Datos y privacidad

Todo se queda en el ordenador. `data/` contiene la base de datos, las versiones de los datasets, los registros (incluido `logs/pygmalion-hoard.log`, el registro rotativo de la propia aplicación, que es lo que queda cuando se arranca sin consola, por ejemplo con `pythonw`), el token de MCP y `secrets.env` (el token de Hugging Face, con permisos solo para el propietario, que nunca se muestra ni se anota en registros). La carpeta de trabajo (`paths.work`, por defecto `data/work`) contiene `hf/` (modelos descargados), `outputs/`, `jobs/` y `calib/`. El único uso de red es Hugging Face (búsqueda y descarga, cuando lo pides), el hub de la familia y Galton en este ordenador. Los archivos importados y lo que genera un modelo se tratan como no fiables: de los adaptadores solo se leen safetensors y JSON, y el texto de los datasets nunca se ejecuta.

## Desarrollo

```bash
python -m pytest -q                  # toda la batería; sin red, sin GPU, sin torch real
python scripts/gen_api_doc.py        # regenerar docs/API.md tras cambiar una herramienta
npx vite build                       # reconstruir la interfaz
```

Parar la aplicación es limpio: los carriles de trabajos, el hilo de eventos y los auxiliares de reserva de GPU se paran y se esperan con un tiempo máximo, y la batería comprueba tras cada prueba que no queda ningún hilo de la aplicación en marcha (`tests/test_shutdown.py`).

FastAPI + SQLite (WAL) + un gestor de trabajos de dos carriles; interfaz con React 19 + Vite + Tailwind. Ver [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) y [AGENTS.md](AGENTS.md).

## Licencia

MIT
