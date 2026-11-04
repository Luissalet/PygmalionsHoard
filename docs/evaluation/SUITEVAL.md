# Evaluation of suits

A suit is a reusable work configuration for a kind of engagement, kept in Darwin's Hoard: objective, inputs, the operations it uses (references to capabilities), views, acceptance criteria, an execution recipe and the points where a model has to judge. This module measures whether a suit, or an improvement of it, produces better deliverables than plain tools. It lives in the evaluation engine (`galton_hoard/suiteval/`), page **Trajes** at `/evaluation/#/trajes`, REST under `/evaluation/api/suit-evals`, and the tools `pygmalion_suit_eval_*`.

## Definition

`pygmalion_suit_eval_define` takes the suit slug (as in Darwin), a name, the cases and the candidate improvement.

A case has an `id`, a `title`, the `input` the arms receive (kept for the people who run it) and one or more **checks**, all of which must pass:

| Check | Fields | Passes when |
|---|---|---|
| `file_exists` | `path`, `min_bytes` | the file exists in the deliverable folder (not empty by default) |
| `json_schema` | `path`, `schema` | the JSON file follows the schema (subset: type, enum, const, required, properties, additionalProperties, items, minimum, maximum, minLength, maxLength, pattern, minItems, maxItems; any other keyword is refused when the check is written) |
| `numeric` | `path` + `pointer`, or `value`, `expected`, `tolerance`, `relative` | the number read from a JSON/text file or handed in with the record is within the tolerance |
| `regex` | `pattern`, `path` or `target: answer`, `flags` (`i`, `m`, `s`), `min_matches`, `max_matches`, `forbid` | the text matches the pattern the required number of times (or not at all with `forbid`) |
| `judge` | `rubric` | a verdict was handed in with the record (`judgements: {check_id: {ok, score?, judge?, note?}}`), from a judge model, the logit judge or a person. Until then the case is *waiting*: it counts neither as passed nor as failed |

Paths are relative to the deliverable folder; absolute paths, drive letters, `..` and symbolic links out of the folder are refused. Files are read, never executed.

## Arms and records

Three arms: `generic` (plain tools), `current` (the suit as it is) and `candidate` (the improvement under test). `pygmalion_suit_eval_record` stores, per case, arm and run: the checks run on the deliverable folder / answer text / numbers / judge verdicts, `correction_minutes` (minutes a person spent correcting it; entered by a person and optional, it can be added later by recording again), `total_time_s`, `defects` (counted from the failed checks until a number is entered) and `tokens` when the harness reports them. Several runs of the same case and arm are averaged into one value per case.

## The case used for building

Judging an improvement on the case it was tuned on measures memory. A case is **excluded from the score of every arm** when it has `used_for_building: true` or is listed in `candidate.built_from_cases`. Excluded cases keep their records and are listed in the score with the reason. The score warns when the candidate was also run on a build case, and when a candidate has results but no case is marked as used for building (nothing then shows it was not tuned on the judging cases).

## Score

`pygmalion_suit_eval_score`: per arm, the cases scored, the share of deliverables that passed with a Wilson 95 % interval, and for total time, correction minutes, defects and tokens: n, mean, standard deviation, min, max, median and sum over the cases that have a value. Paired comparisons (current/generic, candidate/current, candidate/generic) give better, worse and same cases and an exact two-sided sign test. Few cases (under 3 per arm) are flagged as weak.

## Decision

The score also carries `verdict` with two questions, `candidate_vs_current` (adopt the improvement?) and `current_vs_generic` (does the suit beat plain tools?). Each has a `decision`, the `reasons` as sentences, the `basis` per measure and the `tokens`:

| Decision | When |
|---|---|
| `insufficient` | fewer than 3 cases compare both arms (build cases never count) |
| `adopt` | more cases with an accepted deliverable; on a tie, better in a human measure and worse in none |
| `keep` | fewer accepted deliverables; on a tie, worse in a human measure and better in none, or no difference that matters |
| `tradeoff` | the same number of cases improved and regressed in accepted deliverables, or same deliverables but better in one human measure and worse in another: a person weighs it |

The order is fixed: accepted deliverables (paired by case), then correction minutes (a difference matters from 1 minute or 10 % of the incumbent's mean, whichever is larger), defects (0.5 per case) and total time (1 s or 10 %). Tokens are reported with their difference and `decides: false`: when only the tokens differ the decision is `keep`. With a sign test p above 0.05 the reasons say the decision is provisional. The list of evaluations carries `decisions` and the export to Darwin carries `verdict`.

## Suit library

`pygmalion_suit_library` (`GET /evaluation/api/suit-evals/library`) reads the suits of Darwin's Hoard (`<suiteval.darwin_url>/api/suits`; one suit per kind of engagement, with its backing state) and puts next to each the evaluations kept here and the decisions of the latest. Suits evaluated here but not in Darwin are listed as *only here*; when Darwin does not answer, only those are listed and the reason is given. Darwin stays the owner of the suits: this reads them, it never writes them. The **Trajes** page shows the library above the evaluations, with *Open evaluation* and *Evaluate this suit* (fills the definition with the suit's id).

## Export to Darwin's Hoard

`pygmalion_suit_eval_export` (`mode: auto|api|file`) builds the document `pygmalion-suit-eval/1`:

```
{ "format": "pygmalion-suit-eval/1", "suit": "<slug>", "evaluation_id": "se_…", "name": "…", "generated_at": 0,
  "arms": ["generic","current","candidate"], "candidate": {"description": "", "built_from_cases": []},
  "cases_total": 4, "cases_scored": 3, "excluded": [{"case": "k1", "reason": "used_for_building"}],
  "aggregate": { "<arm>": { "cases": 3, "success_rate": 0.67, "success_ci95": [0.2, 0.93], "time_s": {…}, "correction_minutes": {…}, "defects": {…}, "tokens": {…} } },
  "paired": {…}, "per_case": […], "warnings": [{"key": "…", "text": "…"}] }
```

With `auto` it posts to `<suiteval.darwin_url>/api/evaluations` (default `http://127.0.0.1:5233`); if Darwin does not answer or refuses, it writes `<suiteval.export_folder or data/exports/suit-evals>/<suit>--<evaluation id>.json`, which Darwin imports with `darwin_suit_import_eval` (`path`). Nothing is exported while no case outside the building set has a result. In Darwin the suit then reads **evaluated**, with the backing state it had at that moment next to it; an evaluation never says that the suit's capabilities exist.

## Limits

- Judge verdicts are handed in; this module does not call a judge model by itself. Use `logit_judge_run` or a panel and pass the outcome per check.
- The statistics describe the cases that were run. With a handful of cases the intervals are wide and the sign test has little power: read them, do not just quote the rates.
- Correction minutes are only as good as the person who entered them.
