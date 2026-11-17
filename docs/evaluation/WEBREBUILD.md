# Website rebuild: models rebuild a site they can only browse

A model gets a reference website it can only look at through a browser and has to rebuild it as its own site. Both sites then go through
the same scripted cases and the rebuild is scored: a case passes when its functional checks pass and every visual checkpoint reaches a
normalized SSIM of 0.99; a site's score is the share of its cases that pass. The scorer is the family's `hoard_link.webfidelity`, the same
one the web-design studio uses for the fidelity of its clones.

## Sites

The catalogue (`webrebuild_catalog`, `galton_hoard/webrebuild/sites/`) has four small local sites for a quick pass, each with its cases in
`sites/tasks.json`:

- `ficha`: a product page with a header, a picture, a price and a button that adds to the basket.
- `precios`: three plans with a monthly/yearly switch that changes the prices.
- `tareas`: a to-do list: add, mark as done, filter all/pending/done, a counter.
- `blog`: an index of notes with links to each article and an article page with a link back.

Any other reference that is already running is given by URL (`urls`: `{url, title, routes | cases}`): without cases, one picture of each
route is compared.

## Cases

A case is a list of steps (see `hoard_link/webfidelity` in the hub's docs): `goto`, `click`, `fill`, `press`, `select`, `scroll`, `wait`;
the checks `expect_text`, `expect_no_text`, `expect_url`, `expect_count`, `expect_value`; and `checkpoint`, a picture of the viewport. Every
case runs at each width (1280 and 390 px by default) in a fresh browser context with transitions and animations off, a fixed clock and
seeded random numbers. The reference goes first: a case whose own checks fail on the reference is left out, with a warning. A checkpoint of
the rebuild is paired with the reference's by route, width and name; a missing one fails its case.

Normalized SSIM: RGB SSIM over 7x7 windows, rescaled against a flat picture of the reference's median colour, so a blank page does not score
well against a mostly white reference. For every checkpoint under the threshold the zones that differ most are outlined and numbered on the
rebuild's picture.

## Who rebuilds

`models` takes anything Galton can serve (`models_list`: the Sparks' model and other machines of the network, Ollama, a local server, a
GGUF, or `{kind: "server", url, model}`); it needs tool calling (a server that refuses `tools` gets them described in the prompt). One session
per model rebuilds every site in turn. The model's tools:

- `browse`, `click`, `fill`, `press`, `back`: act on the reference at 1280 or 390 px and get back a text picture of the page: every visible
  element with its tag, selector, position and size, text, font, colours, border, radius, padding and flex/grid layout. Never its source.
- `write_file`, `read_file`, `list_files`: the files of its own site (HTML, CSS, JS, SVG, JSON; relative paths, up to 40 files).
- `preview`: its own page, seen the same way.
- `finish`.

The loop stops at `finish`, when the model stops calling tools twice, or after `max_steps` (30). Old snapshots are cut to their first line
so the context does not grow without end. Every request leaving the site is blocked.

## What is measured

Per model and site: the score (share of cases passed), cases passed, the mean visual score, the share of checks that pass, the steps,
tokens in and out, the time to build and to score, and the cost at the price per million tokens given in the call or in
`briefbench.prices` (unknown, never zero, without a price). Per model, the means over its sites and the best, fastest and cheapest. A site
whose model never answered is `none`, left out of the means.

## WebsiteBench

`webrebuild_websitebench` keeps the public part of WebsiteBench (Apache-2.0, see `provenance/websitebench/NOTICE`) in the data folder:
`action=import` downloads its release index, licence and notice and, for the tasks named (or `every`), the instruction, the task file
and the case manifest (how many hidden cases of each kind and tier). `action=list` and `action=instruction` read them. The hidden cases
and the reference sites live in that benchmark's containers: a full run goes through the agent bench, benchmark `websitebench`
(Docker and the trial runner, hours per task).

## Files and screens

`<data>/webrebuild/<run>/run.json` holds the run; `<run>/<site>/reference/` the reference pictures; `<run>/<site>/<slot>/site/` the files the
model wrote and `shots/` its pictures and the diff pictures (`diff--*.png`), served by `/api/webrebuild/<run>/<site>/<slot>/<file>`.
`webrebuild_score` scores a running site against a running reference without any model.

The screen is Webs: choose sites and models (with prices), see the plan, follow the run, read the table per model and per site, and open a
rebuild to see each case with its checks and its reference, rebuild and difference pictures side by side. Settings: `webrebuild.browser_python`
(a Python with Playwright and Chromium; empty: the one of Briefs or Video, else found) and `webrebuild.folder`.
