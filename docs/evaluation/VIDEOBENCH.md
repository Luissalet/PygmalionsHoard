# Video bank: how well a model writes an animation as code

A model gets a brief for an animation and writes one self-contained HTML page. A browser draws the frames of that page, automatic checks say what the pictures
do, and a jury that did not write the page scores them. The point is to compare models on a task where the result is something you can look at, not to
produce videos.

## The briefs

139 briefs (`galton_hoard/videobench/data/prompts.json`), taken from the public list awesome-opus5-5-videos (MIT; see
[provenance/videobench/NOTICE](../../provenance/videobench/NOTICE)): those whose prompt is not partial and has more than 400 characters. Each one keeps its
author, the link to the author and the link to the original post, and the screen and the tools show them. The videos and the prompts belong to their authors. A brief is
marked `standalone` (103 of 139) when it does not mention an attached file, a link or an image, which a model cannot use; a run samples only those unless you
ask for others. Categories: `motion` 72, `3d` 27, `interactive` 24, `explainer` 16.

A sample is deterministic: the same `seed`, filters and `count` give the same briefs.

## The contract

The model is asked for one HTML document (`contract.SYSTEM`): no network (no CDN, external script, font, image or video, everything inline), a 1280x720 stage,
`window.DURATION` (the length in seconds, at most 30; 10 when the brief names none) and `window.seek(t)`, which draws exactly the frame at time `t` and is
deterministic. This is what lets a renderer take any frame in any order. The animation is silent; it may also play itself with a `requestAnimationFrame` loop. A page that
is not an HTML document, or that has no `window.seek`, is an item that did not render.

## The three phases

1. **Writing.** The model writes a page per brief (`max_tokens` 16000, `timeout_s` 900, temperature 0.7). A reply that is cut off is read as far as it goes.
2. **Rendering** (no model). A Python with Playwright runs `render_worker.py` in a subprocess: it blocks every network request (counted, not hidden), turns the
   page's own `requestAnimationFrame` off, takes `frames` (12) pictures at `t` spread over the length, and checks determinism by drawing the middle, then 0, then the
   middle again and comparing the pictures. Then `render.measure` computes blank frames, the mean change between frames, still stretches and the largest change of
   any frame from the first (a loop ends where it began, so the last frame is not the measure), and `contact_sheet` makes a 4x3 sheet of 1920x810. A page is
   **usable** when it moves, is not blank (at most a third of the frames), is deterministic and raised no page error.
3. **Jury.** A fresh session, which did not write the page, gets the brief, the automatic checks (as numbers, not as a verdict) and the contact sheet plus two
   larger frames, and returns JSON with six scores from 1 to 10 (`style_fidelity` is fidelity to the brief), a summary, fixes ordered by impact (time, problem, fix)
   and critical issues. A jury model without `vision` judges from the checks and the source of the page and the item says so (`jury_mode: text`). The rubric
   is the one of the mg-styles-15 review prompt (MIT) minus `sound_sync`, because the pages are silent. A jury reply with no scores is asked once more; if it still has
   none, the item is `None`.

The writer and the jury can be different models (`jury`); with none given the same model judges, and it tends to be kinder to its own pages: the plan and the result say so.

None is not zero: an item that never reached a phase (server down, no HTML, no render, no readable jury reply) is `none` or `unrendered` with its reason and is left out
of the means. After four requests in a row with no answer the rest of the run is stored as `None`.

## Run it

From the app: screen Vídeo (`#/videos`): choose the writing model, the jury, how many briefs, the category and the seed; the plan line says how many pages will be written and
whether the jury sees images; the run page shows the three phases, the summary with the means per dimension and, per brief, the contact sheet, the page itself playing with a time
slider (in a sandboxed frame without network), the jury's critical issues and fixes and the brief with its author and post. Tools: `videobench_catalog`, `videobench_prompt`,
`videobench_start` (`dry_run` shows the plan), `videobench_status`, `videobench_result`, `videobench_item` (with `html: true` for the page), `videobench_cancel`, `videobench_list`.

The renderer needs a Python with Playwright and Chromium (`pip install playwright` and `playwright install chromium`). Galton tries its own interpreter and `python`, `python3` and
`py`; set `videobench.browser_python` (Ajustes > Banco de vídeo) to name another. The results are in `<data>/evaluation/videobench/<run>/<brief>/` (`index.html`, `sheet.png`,
`frames/`). `videobench.folder` moves them.

Files are served to the app itself only (same origin), by a whitelist of names; the page is served with a CSP that allows no network and a sandbox without `allow-same-origin`.

## A real measurement

[docs/evaluation/measurements/video-bank-deepseek-v4.1-flash](measurements/video-bank-deepseek-v4.1-flash/README.md): DeepSeek V4.1 Flash writes ten briefs (seed 0) and the same model
judges. All ten pages rendered and were deterministic; seven were usable; nine were scored, with a mean of 4.2 of 10 (concept 3.3, motion 3.8, design 4.7, technique 5.1); one jury reply
was unreadable (`None`). Each folder has the page the model wrote and the contact sheet the jury saw.

## Limits

- The jury is a model: its scores are a judgement, not a measurement, they depend on the jury model and on the sample, and the same model as writer and jury is kinder. Compare scores
  only between runs with the same jury.
- The same brief scored 4.3 in the ten-brief run and 3.8 when it was run alone through the screen (a new page was written each time): one page and one jury reply are noisy. Use several
  briefs and seeds, and look at the means, not at one score.
- A contact sheet of 12 frames cannot show everything: a short effect between two frames is missed, and fine motion quality is judged from the spacing of the frames.
- The pages are silent, so sound is not scored. Briefs that ask for a library, a file, an account or a device are still asked; the model must leave out what it cannot do, and the
  jury judges against what a silent offline page can show.
- Rendering is Chromium at 1280x720; heavy 3D and shader pages can hit the render timeout (`render_timeout_s`, 240 s) and are then `unrendered` with that reason.
- Ten briefs are a smoke measurement. Use `count` and several seeds before ranking models.
