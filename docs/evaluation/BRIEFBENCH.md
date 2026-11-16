# Brief bench: the same brief to several models

One brief goes to several models. The bench times every answer, counts its tokens, estimates what they would have cost at the price you give, renders the answer when it
is a page, and puts everything side by side. An optional judge that did not write the answers scores them without knowing whose they are.

## Briefs

A brief is a prompt and a kind:

- `page`: the answer is one self-contained HTML document. It is drawn in a headless browser; the grid shows the picture, the screen plays the page in a sandbox.
- `text`: the answer is read as it comes (an explanation, a code review, a translation).

The catalogue (`briefbench_catalog`, `galton_hoard/briefbench/briefs.py`) has a Three.js dragon over a castle, a solar-system orrery on a canvas, a pricing page, a
dashboard drawn with inline SVG, a Snake game, a Spanish explanation and a bug hunt. Any other brief is written in the call (`brief`, or `briefs` for several, each with its
own `kind`, `title`, `libraries` and `criteria`) and gets the id `own-1`, `own-2`...

## Who answers

`models` takes anything Galton can serve: a model of `models_list` (the language models of the Sparks and other machines of the network, Ollama, a local server, a GGUF) or
a spec `{kind: "server", url, model}`. A command-line agent that has its own session joins as `cli:<name>`: write it in the setting `briefbench.cli`, one line `name = command`,
where `{prompt}`, `{prompt_file}` and `{workdir}` are replaced in the arguments (and exported as `BRIEFBENCH_PROMPT`, `BRIEFBENCH_PROMPT_FILE`, `BRIEFBENCH_WORKDIR`). What the command
prints is the answer; for a page brief, an `index.html` it leaves in the folder is the answer when the output holds no HTML document. Nothing is configured by default, a line
starting with `#` is off, and the bench never starts an agent that is not written there. The agent counts and bills its own tokens, so those are unknown for its answers.

One session per model answers all the briefs, one model after another, so a model is loaded once. A model that cannot be used leaves its answers as `none` and the others go on.

## What is measured

Per answer: the time of the call (`latency_s`), the time to the first token and the decode speed when the server reports them, the tokens in and out, the tokens per second,
and the cost: tokens times the price per million tokens, from the call (`prices`: `{"<model>": {"input": 3, "output": 15}}`, a single number for both) or from the setting
`briefbench.prices` (one line `model = input, output`), matched by id or name. A model without a price has an unknown cost (`None`, shown `—`), never a zero one, and the
summary says how many answers had no price. The currency label is the setting `briefbench.currency`.

For a page: whether the browser drew it, errors of the page and the console, requests the page tried to make (all blocked and counted), whether it is blank, and whether it moves by itself
(two pictures `motion_ms` apart, `settle_ms` after loading). WebGL runs on the software rasteriser of the browser, so a 3D page never needs the graphics card.

A page may lean on an offline library. Put the file in the folder of the setting `briefbench.libraries` (`three.min.js`, `d3.min.js`...) and name the library in the brief's `libraries`: the
model is told to load it with `<script src="lib/<file>"></script>`, and the file is put into the page when it is drawn or played, so what the model wrote stays as written. A named library
without a file is reported in the plan, and the model is told to write what it needs by hand.

## The judge

`judge` names a model that scores every answer in a fresh session on four dimensions from 1 to 10: `fidelity`, `craft`, `polish` and `technical`. It never gets a model name. With vision it looks
at the two pictures of a page; without it, it reads the automatic facts and the page source (the run says which, `judge_mode`). A text brief is judged on its text. The judge is a model:
its scores are a judgement, not a measurement, and a model that is also one of the answerers tends to be kinder to itself (`same_model_judges`). A judge reply without scores is asked
again once, then left as `none`; the answer stays an answer.

## Reading the result

`briefbench_result` gives, per model, the answers that worked, the mean time, tokens, tokens per second, total cost, mean score and who is best, fastest and cheapest (only when there is
a single one); per brief, a table of rows and `grid_url`, a PNG with every model's render side by side with its figures. An answer that never arrived (`none`) or a page that did not draw
(`unrendered`) is in the table with its reason and is left out of the means; it is never a zero.

## Files and screens

Files go to `<data>/briefbench/<run>/<brief>/<slot>/` (`index.html`, `shot.png`, `shot2.png`, `answer.txt`) and `<run>/<brief>/grid.png`. `api/briefs.py` serves them with a whitelist of names:
`/api/briefbench/<run>/<brief>/grid.png`, `/api/briefbench/<run>/<brief>/<slot>/shot.png` and `.../play.html` (the page under a sandbox with no origin, no network and no navigation).

The screen is Encargos: choose briefs, models (with their prices) and a judge, see the plan, and open a run to compare the grid, sort it by score, time or cost and open any answer.

## Tools

`briefbench_catalog`, `briefbench_brief`, `briefbench_start` (`dry_run` shows the plan; `wait_s` waits for the result), `briefbench_status`, `briefbench_result`, `briefbench_item`, `briefbench_cancel`,
`briefbench_list`, `briefbench_delete` (`confirm=true`). Settings: `briefbench.browser_python` (falls back to `videobench.browser_python`, then to a Python that has Playwright),
`briefbench.folder`, `briefbench.libraries`, `briefbench.prices`, `briefbench.currency`, `briefbench.cli`, `briefbench.cli_timeout_s`.
