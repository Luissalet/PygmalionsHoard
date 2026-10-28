# Video bank: deepseek-v4.1-flash writes, the same model judges

Measured on 2026-10-28, run `vb_01m622aw0885rdq8c5v4eywpcg`. 10 briefs sampled with seed 0 among the 103 that need no attached file; the model wrote one self-contained HTML page per brief (max 16000 tokens, temperature 0.7), a browser rendered 12 frames of each at 1280x720 with the network blocked, and a jury in a fresh session (the same model, which sees images) scored the contact sheet on six dimensions (sound is not scored: the pages are silent). The method is in `docs/evaluation/VIDEOBENCH.md`; files per brief: `index.html` (what the model wrote), `sheet.jpg` (the 12 frames the jury saw).

The jury is the same model that wrote the pages, which tends to be kinder to them. The scores are a judgement, not a measurement.

| measure | value |
|---|---|
| briefs / pages that rendered | 10 / 10 |
| deterministic (`seek(t)` draws the same picture each time) | 100 % |
| usable (moves, not blank, deterministic, no page errors) | 7 of 10 (70 %) |
| judged / without a jury result (None, left out of the mean) | 9 / 1 |
| mean jury score (1 to 10) | 4.24 |
| mean output tokens / writing time per page | 8349 / 108.9 s |

| dimension | mean |
|---|---|
| style_fidelity | 4.44 |
| concept_wow | 3.33 |
| motion_craft | 3.78 |
| design_typography | 4.67 |
| finish_texture | 4.11 |
| technical | 5.11 |

Per brief (author and post are in `result.json`; every brief belongs to its author):

| brief | category | outcome | moves | usable | mean | style_fidelity | concept_wow | motion_craft | design_typography | finish_texture | technical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [pluvio9yte-662953](https://x.com/Pluvio9yte/status/2107087191738662953) | motion | done | True | True | 4.33 | 4.0 | 3.0 | 4.0 | 5.0 | 4.0 | 6.0 |
| [hsanat-012699](https://x.com/hSanat/status/2103358205091012699) | motion | done | True | True | 5.0 | 5.0 | 4.0 | 4.0 | 6.0 | 5.0 | 6.0 |
| [hicallmechai-878304](https://x.com/hiCallMeChai/status/2103600312296878304) | motion | done | True | True | 4.83 | 5.0 | 4.0 | 4.0 | 5.0 | 5.0 | 6.0 |
| [dzhohola-249003](https://x.com/dzhohola/status/2103605737595249003) | motion | done | True | True | 4.67 | 5.0 | 4.0 | 5.0 | 5.0 | 5.0 | 4.0 |
| [dale-vaz-879074](https://x.com/dale_vaz/status/2103290592679879074) | motion | done | True | False | 3.67 | 4.0 | 3.0 | 3.0 | 5.0 | 4.0 | 3.0 |
| [okapi-fukugyo-166351](https://x.com/okapi_fukugyo/status/2103586082336166351) | motion | done | True | True | 3.67 | 4.0 | 3.0 | 4.0 | 4.0 | 4.0 | 3.0 |
| [thejohnnyhuy-282154](https://x.com/thejohnnyhuy/status/2104149060588282154) | motion | done | True | True | 3.17 | 3.0 | 2.0 | 3.0 | 3.0 | 2.0 | 6.0 |
| [zacxbt-944604](https://x.com/zacxbt/status/2103808699466944604) | interactive | done | False | False | 5.17 | 6.0 | 4.0 | 4.0 | 5.0 | 5.0 | 7.0 |
| [twoclipping-041659](https://x.com/twoclipping/status/2107822293653041659) | interactive | none (jury_unreadable) | True | True | - | - | - | - | - | - | - |
| [aichrislee-922082](https://x.com/aichrislee/status/2107260223907922082) | interactive | done | False | False | 3.67 | 4.0 | 3.0 | 3.0 | 4.0 | 3.0 | 5.0 |

What the jury said of each page:

- **pluvio9yte-662953** (Pluvio9yte): The page is a technically sound particle system with a working seek and a clean HUD, but it never delivers the brief's core promise: a source image decomposed into depth, subject and colour, then rebuilt in contrasting styles with beat-locked feedback. Every frame is an undifferentiated cloud of dots on black, so the 'before → particle' reveal, the subject silhouette, the depth push-in and the hero moment are all absent.
- **hsanat-012699** (hSanat): A competent but generic dark-SaaS template: logo, dashboard, feature cards, stats, quote, CTA, in that order, with no hook, no hero moment and no real product story. The typography is clean and the palette is consistent, but the motion is mostly fades and the 15.3s stat collision is a visible failure.
- **hicallmechai-878304** (hiCallMeChai): The state list is all there and the loop closes, but the camera never zooms, so every state sits as a tiny chip in a vast empty canvas — the opposite of the brief's 'each state fills the frame'. The morphs read as small, timid and mostly unreadable at contact-sheet scale, and the hero moments (volume stretch past max, liquid tab indicator, chart draw) are invisible.
- **dzhohola-249003** (dzhohola): The morph chain is legible and the black-on-warm-gray palette is on brief, but the piece reads as a storyboard of separate widgets rather than one shape never cut: the container jumps size and position between states instead of morphing, the camera never zooms to fill frame, and the cursor is a tiny afterthought. The loop closes numerically but the last frame is a bare pill with no content, so the return to 'Get started' is a pop, not a resolution.
- **dale-vaz-879074** (dale_vaz): The desktop UI itself is competently drawn — the option chain, watchlist sparklines and chart panel read as a real terminal — but the film around it is mostly empty black: 5 blank frames, 3 still steps, and a first-to-last change of 0.0, meaning the piece begins and ends on nothing. The brief's two hero beats (zoom out to a point in space, zoom back in to a landscape mobile split-screen ending on 'Single Screen Trading on SAHI') are absent or invisible, so the whole thing reads as a UI mockup with a slow fade rather than a directed film.
- **okapi-fukugyo-166351** (okapi_fukugyo): The single-morphing-shape premise is legible and the palette is right, but the piece is mostly empty warm-grey field with a small black blob in the middle: the camera never zooms to fill the frame, the states are unreadable at contact-sheet scale, and the text layer is a static dump of every label at every time. It reads as a proof-of-concept for the morph engine, not as a Dribbble-grade screen-motion piece.
- **thejohnnyhuy-282154** (thejohnnyhuy): This is a soft-focus blob field drifting over a static iris polygon with a hue-cycling gradient behind it; nothing in it reads as liquid glass, goo, or an Apple keynote. The wordmark arrives as a blurred white sans sitting on top of the blobs rather than being formed by them, so the one hero moment the brief demands is missed.
- **zacxbt-944604** (zacxbt): A competent, correct pixel-art wizard that hits most of the brief's technical clauses (integer scaling, palette, pooled particles, deterministic seek) but reads as a static sprite with a sparkle emitter rather than an animated performance. The wizard barely moves, the 'cast' has no projectile, no burst, no shake, and the composition is a small figure floating in a large empty field with a flat two-band sky.
- **twoclipping-041659** (twoclipping): (no jury result)
- **aichrislee-922082** (aichrislee): The page renders and seeks deterministically, but the 'V8 engine' reads as a stack of flat grey slabs with a blue comb on top; the V-angle, cylinder banks, pistons, rods, crank, cams and valves are not legible as mechanical forms. Nothing in the 12 seconds demonstrates the explode, the labels, the hover state or the orbit, so the brief's core interactions are unproven and the hero moment is absent.

Reading it:

- All ten pages rendered and all ten were deterministic: the `window.seek(t)` contract is within reach of the model. What it does not reach is craft: the mean is 4.2 of 10 (5 is a competent student), with the concept and the motion lowest (3.3 and 3.8) and the technique highest (5.1).
- Loops end where they began, so "moves" looks at how far any frame gets from the first, not at the last frame.
- One page could not be scored: the jury answered twice without scores. It is a `None`, not a zero.
- The first brief was run again alone through the screen (a new page, the same jury model) and scored 3.8 against 4.3: one page and one jury reply are noisy.
- Ten briefs and one model: a smoke measurement, not a ranking. The first brief of the sample describes a pipeline that starts from the author's footage; the page can only imitate it.
