# Language bench: deepseek-v4.1-flash, Spanish and English

Measured on 2026-10-28 with `scripts/evaluation/harnessbench_eval.py --langs es,en --k 2 --max-tokens 1536` against the Sparks endpoint
(DeepSeek V4.1 Flash). 27 tasks x 2 languages x 2 samples = 54 trials per cell, temperature 0.7. The method and how to read it are in
`docs/evaluation/HARNESSBENCH.md`. The full report, with every trial and its trace, is `language-bench-deepseek-v4.1-flash.json`.

| harness | lang | n | pass@1 | pass^k | calls mean | calls vs en | consistency | empty searches | arg mismatch | format fails |
|---|---|---|---|---|---|---|---|---|---|---|
| native_tools | es | 54 | 92.6 % | 92.6 % | 2.04 | 0.94 | 99.8 % | 7.7 % | 0.0 % | 0.0 % |
| native_tools | en | 54 | 94.4 % | 92.6 % | 2.17 | - | 100.0 % | 0.0 % | 3.8 % | 0.0 % |
| json_text | es | 54 | 77.8 % | 74.1 % | 1.94 | 0.955 | 95.2 % | 3.9 % | 0.0 % | 0.0 % |
| json_text | en | 54 | 92.6 % | 88.9 % | 2.04 | - | 100.0 % | 0.0 % | 0.0 % | 0.0 % |
| xml_text | es | 54 | 44.4 % | 37.0 % | 0.72 | 1.5 | 100.0 % | 3.3 % | 0.0 % | 51.8 % |
| xml_text | en | 54 | 35.2 % | 22.2 % | 0.48 | - | 100.0 % | 0.0 % | 5.3 % | 64.8 % |
| react_text | es | 54 | 87.0 % | 85.2 % | 2.43 | 0.978 | 99.7 % | 5.5 % | 0.0 % | 0.0 % |
| react_text | en | 54 | 92.6 % | 85.2 % | 2.48 | - | 100.0 % | 1.9 % | 0.0 % | 0.0 % |

Over the four harnesses: Spanish 75.5 % pass@1, English 78.7 % (3.2 points); language consistency 98.7 % / 100 %; empty searches 5.5 % / 0.5 %;
arguments in another language than the data 0 % / 2.0 %.

Effects: the language moves pass@1 by 7.9 points on average (per harness, absolute), the harness by 53.7 points; the model effect cannot be told with one model.

Notes:

- `xml_text` is mostly a format failure: the model writes its own tool-call markup (DSML) instead of the XML the harness asks for. Before the harnesses recognised that
  markup (and answered it with a format error), the same cell scored 9.3 % (es) and 13.0 % (en) because those calls were read as wrong final answers.
- The empty searches in Spanish come from the model translating a closed value of the catalogue (`in_transit` to `en tránsito`).
- One model and 54 trials per cell: intervals are wide. The figures do not compare with other benchmarks or with a model judge.
