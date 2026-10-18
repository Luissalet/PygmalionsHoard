# Training and evaluation fusion

Pygmalion owns the native Galton package and source UI, licensed MIT; source revision/repository are recorded in `provenance/galton/source.json`, with original licence and NOTICE. No runtime import, server process or socket depends on the former checkout. The module lifetime/stream dispatcher derives from the family's reviewed Talos implementation and is adapted here. Package discovery includes both owned engines and their static/suite assets.

## Lifetimes

`python -m pygmalion_hoard` launches `fusion.app:create_app`. Import/listing the MCP catalogue reads shipped JSON only. The shell opens neither domain database until its module is enabled. Training, evaluation and automatic evaluation are independently persisted; empty `enabled` is the default. Automatic evaluation is a live scheduler pause overlay, leaving prior settings intact. Manual evaluation jobs remain available with this switch OFF. The overlay also blocks automatic discovery and housekeeping. Historical pending work is not resumed: training keeps an in-memory set of old queued IDs (explicit resume releases the hold); evaluation does not requeue runs/matrices/campaigns on open.

The native `NativeGalton` facade implements the original evaluator's call/available/clock/sleep contract and invokes the incorporated tool handlers synchronously in process. It never uses the Hub proxy, old URL/token or an HTTP socket. Native calls retain the `pygmalion` caller on evaluation runs and translate evaluation errors into the training evaluator’s expected error type, preserving recoverable comparison/regression handling. Existing standalone transport code remains for isolated source regression, outside the production wiring. GPU coordination and model-provider implementations are retained.

OFF is saved before waiting. Native API requests share a lifetime and tracked streams are cancelled on stop; regular requests drain. Native calls from training workers acquire an active-call count; closing evaluation refuses new calls and drains the old ones before its database closes. Services cancel/join job lanes, matrices and agent campaigns before database close. Lifespan shutdown runs in a worker thread so the ASGI event loop stays responsive. If a worker exceeds its join deadline, the database remains open and shutdown stays pending; another disable retries cleanup rather than claiming OFF. No shared model server is killed.

## Data

`data/modules.json`: `{ "enabled": [], "data_paths": { "training": "<original Pygmalion data>", "evaluation": "<original Galton data>" } }`. Training defaults to the current data folder. Evaluation defaults to `data/evaluation` for clean installs. Existing installations select the original Galton directory explicitly using `scripts/adopt_evaluation.py`. This writes only the map, never opens/copies/migrates an engine database. On explicit evaluation activation, the original data folder’s parent `.env` is read in place for Galton secrets, without copying or logging it; the process environment still takes precedence. Tokens, datasets, artifacts, lineages, model files, results and suites remain in place. Once a user enables an engine, its normal source migrations and interrupted-run bookkeeping apply.

## Surfaces

- `/`: Functions; `/api/modules`, `/api/modules/{id}`, `/api/modules/stop-all`.
- `/training/`: native training SPA/API; `/evaluation/`: native evaluation SPA/API. Same origin, one ASGI process, no iframes. Source clients are compiled with explicit bases and API prefixes.
- `/capture/...`: compatibility route into training, OFF when training is disabled.
- Root `/api/agent/tools` and `/api/agent/call`: 117 tools, including independent-module controls.
- The three conflicting evaluation tool names gain `evaluation_`: settings_get, settings_set, gpu_status. Their schemas and native names are unchanged inside `/evaluation/api/agent/`.
- Existing native source API documentation: `API.md` (50 training tools), `evaluation/API.md` (65 evaluation tools). The committed combined schema lives in `pygmalion_hoard/fusion/catalog.json`; tests compare every entry to both original catalogues.

## Verification

Run `python -m pytest tests evaluation_tests fusion_tests --import-mode=importlib`; fake workers, invented data, no GPU/real model. Build both native clients with `npm run build`; check both API docs generators. Browser verification must inspect Functions on desktop/mobile and navigate native pages after explicit enabling. A source regression does not certify inference quality of a real model.
