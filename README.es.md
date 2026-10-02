# Pygmalion's Hoard

Un estudio local para adaptar modelos de lenguaje. Prepara datasets, entrena adaptadores LoRA y QLoRA, fusiona adaptadores y modelos enteros, amplía la ventana de contexto, convierte a GGUF, cuantiza con una matriz de importancia calculada sobre tus propios textos, mide el resultado frente a su origen y publica el mejor en Ollama o como servidor llama.cpp. Cada archivo que produce conserva su linaje, y la receta que lo creó se puede exportar y repetir.

Forma parte de la familia de apps locales Hoard: funciona en tu ordenador, guarda sus datos en `data/`, se usa sola desde el navegador y puede manejarla un asistente por MCP.

[English version](README.md)

## Qué hace

- **Datasets.** Fuentes: archivos sueltos, una carpeta, JSONL y CSV pegados o en archivo, herramientas de otras apps Hoard (a través del hub de la familia) y ejemplos sintéticos que escribe un modelo profesor a partir de tus textos. Las operaciones se aplican en orden: quitar duplicados exactos y casi duplicados (MinHash sobre n-gramas de 5), filtros de longitud e idioma, y enmascarar o descartar datos personales (correos, teléfonos, IBAN, tarjetas, DNI/NIE). El reparto entre entrenamiento y evaluación usa semilla. Las versiones son inmutables; revisar registros (aceptar, rechazar, editar) crea una versión nueva. Estadísticas: registros, tokens, histograma de longitudes, reparto de roles e idiomas.
- **Entrenamiento.** LoRA y QLoRA (4 bits NF4, cálculo en bfloat16) sobre un modelo de lenguaje causal, con pérdida solo sobre los turnos del asistente en datos de chat, empaquetado en texto plano, evaluación periódica, puntos de control para reanudar, calendario coseno y semilla fija. Antes de empezar, `train_plan` estima la memoria de GPU con su fórmula, estima el tiempo e indica qué GPU permitida cabe. El tiempo sale de los tokens que hay que procesar (los registros que recorren los pasos x el registro medio, cortado en `seq_len`, x relleno) divididos por un modelo de velocidad: 1 600 tokens/s por cada mil millones de parámetros en QLoRA sobre una tarjeta de 16 GB de la serie 50, 1,6 veces más con LoRA normal, escalado por la velocidad de la tarjeta y acotado para modelos muy pequeños, más un arranque y un coste por paso. Las constantes son aproximadas y el plan muestra la fórmula con sus números. Cuando el trabajo arranca, los tokens/s medidos y el tiempo restante real sustituyen a la estimación (la vista del trabajo dice cuál muestra), y un trabajo terminado muestra cuánto duró frente a lo estimado.
- **Fusiones.** Un adaptador en su base (tensor a tensor, sin cargar el modelo entero), o varios modelos de la misma arquitectura con lineal, SLERP, TIES o DARE. Antes se comprueba la compatibilidad.
- **Ampliación de contexto.** Una copia del modelo con YaRN para 2, 4 u 8 veces el contexto, una tabla de memoria de la caché KV por longitud a partir de los metadatos del GGUF y los resultados de la aguja en un pajar por longitud.
- **GGUF y cuantización.** Conversión con el llama.cpp instalado (antes se comprueba la arquitectura), matriz de importancia en una GPU reservada, varios tipos de cuantización de una vez (Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, Q3_K_M, IQ3_M) y perplejidad con su error. La perplejidad se guarda en el archivo medido con su error, texto, contexto y fragmentos; cuando una cadena cuantiza a varios tipos con perplejidad activada, se mide igual el archivo sin cuantizar y cada cuantización, y el archivo propio del modelo base cuando existe o se construye para la evaluación. Linaje y Cuantizar los muestran juntos, con el cambio respecto al archivo padre (lo que costó la cuantización) y respecto al del base del mismo tipo (lo que hizo el ajuste).
- **Evaluación.** Cada resultado GGUF o de Ollama se compara a través de Galton's Hoard (por el hub, o directamente si no está), y el veredicto (mejor, peor, sin diferencia clara) con su diferencia e intervalo se guarda en el resultado junto con el tipo de referencia en que se apoya. Con qué se compara depende de lo que se pregunte. Las intenciones que preguntan si un entrenamiento ayudó o perjudicó (`dataset`, `style`, `writing`, `code`, `general`, `smoke`) comparan con el **modelo base del que partió el entrenamiento, escrito como el resultado**: el mismo tipo de cuantización y, si el resultado usó una matriz de importancia, una matriz calculada con el mismo texto de calibración. El antecesor GGUF más cercano de un archivo ajustado es el f16 del mismo ajuste, que mediría la cuantización y no el entrenamiento. Si el base aún no tiene un archivo así, `evaluate_start` encola los pasos que lo hacen (convertir la carpeta del base, matriz, cuantizar; lo que ya existe se reutiliza) como una sola cadena que termina en la evaluación; los archivos quedan en el linaje bajo el modelo base y la siguiente evaluación los encuentra. `context` compara con el modelo anterior a la extensión de contexto, construido igual. Una cuantización sola (sin entrenamiento detrás; intención `quant`, su valor por defecto) sigue comparándose con el f16 del que salió. `against` manda siempre. `evaluate_plan` (y `artifact_get`, como `evaluate_plan`) dice qué referencia usaría una evaluación y si hay que prepararla antes; el cuadro «Evaluar» de Linaje muestra esa frase. Un resultado cuyo modelo base no está en el linaje se compara con su archivo padre y el veredicto lleva un aviso. Un resultado que desciende de un entrenamiento cuyo dataset reserva registros se mide por defecto con esos registros (`intent: dataset`): los registros reservados de esa versión del dataset se convierten en una suite de Galton `pyg-<dataset>-v<n>-eval` (se crea una vez por versión y se reutiliza; cada registro es un caso con la respuesta de referencia y un comprobador de juez), el padre y el hijo la responden y una suite general pequeña (`rapida`) corre a su lado para detectar regresiones. Se guardan y se muestran los dos veredictos; un resultado general peor detiene la sugerencia de promover. El juez es el modelo propio de Galton (su ajuste `judge.contestant`): sin uno, la evaluación se detiene antes de empezar y dice dónde configurarlo. `galton.eval_cases` limita el número de casos.
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

La interfaz viene compilada en `pygmalion_hoard/static`. Para reconstruirla: `npm install && npx vite build`.

Entorno: `PYGMALION_PORT` (5202), `PYGMALION_DATA_DIR`, `PORT_STRICT=1`, `PYGMALION_SCHEDULER=0` (sin trabajo en segundo plano), `PYGMALION_OFFLINE=1` (sin red), `PYGMALION_WORKERS_DIR`, `PYGMALION_ALLOWED_HOSTS`, `PYGMALION_HTTP_TIMEOUT_S`, `PYGMALION_HF_TOKEN` (o guárdalo en Ajustes). Todo lo demás son ajustes guardados en `data/pygmalion.db`.

## Asistentes (MCP)

`mcp_server.py` es un puente MCP por stdio llamado `pygmalion-hoard`. Nunca abre la base de datos: reenvía cada llamada a la app en marcha con el token de `data/mcp-token`, y arranca la app cuando no responde. `faustus-plugin.json` describe la app, su comprobación de salud y el puente para Faustus y el Hoard Hub.

Herramientas (44): `pygmalion_overview`, `env_check`, `bases_list`, `base_get`, `hf_search`, `base_download`, `datasets_list`, `dataset_get`, `dataset_create`, `dataset_records`, `dataset_review`, `dataset_preview_source`, `dataset_apply`, `dataset_delete`, `train_plan`, `train_start`, `merge_check`, `merge_lora_start`, `merge_models_start`, `ctx_extend_start`, `ctx_fit`, `convert_start`, `quantize_start`, `perplexity_start`, `jobs_list`, `job_get`, `job_cancel`, `job_resume`, `job_delete`, `artifacts_list`, `artifact_get`, `artifact_recipe`, `artifact_update`, `artifact_delete`, `lineage_graph`, `evaluate_plan`, `evaluate_start`, `publish_ollama`, `publish_llama`, `unpublish`, `settings_get`, `settings_set`, `secret_set`, `gpu_status`. Los argumentos están en [docs/API.md](docs/API.md).

Borrar exige `confirm=true`. Permitir una GPU del dueño exige `confirm_reserved=true`, que un asistente no debe pasar salvo que el dueño lo haya dicho.

## Datos y privacidad

Todo se queda en el ordenador. `data/` contiene la base de datos, las versiones de los datasets, los registros (incluido `logs/pygmalion.log`, el registro rotativo de la propia aplicación, que es lo que queda cuando se arranca sin consola, por ejemplo con `pythonw`), el token de MCP y `secrets.env` (el token de Hugging Face, con permisos solo para el propietario, que nunca se muestra ni se anota en registros). La carpeta de trabajo (`paths.work`, por defecto `data/work`) contiene `hf/` (modelos descargados), `outputs/`, `jobs/` y `calib/`. El único uso de red es Hugging Face (búsqueda y descarga, cuando lo pides), el hub de la familia y Galton en este ordenador. Los archivos importados y lo que genera un modelo se tratan como no fiables: de los adaptadores solo se leen safetensors y JSON, y el texto de los datasets nunca se ejecuta.

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
