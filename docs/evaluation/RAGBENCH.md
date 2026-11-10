# RAG pipeline bench

When a retrieval-augmented pipeline answers badly, this bench says which piece failed. It measures the pipeline one piece at a time (the retriever, the reranker, the generator), attributes every failed case to the piece that lost it, and compares two configurations case by case.

Code: `galton_hoard/ragbench/` (`metrics.py`, `attribution.py`, `judge.py`, `pipelines.py`, `generate.py`, `service.py`, `store.py`), REST in `galton_hoard/api/ragbench.py`, page **Tuberías RAG** (`evaluation-client/src/pages/Rag.jsx`), tests in `evaluation_tests/test_ragbench.py`. Tables `rb_sets`, `rb_cases`, `rb_runs`, `rb_results` (migration 13).

## Case sets

A case set is a small corpus plus cases. A case has a question, the reference passages (what must be retrieved to answer it) and the expected answer. A case marked as not answerable has no passage: the right answer is to say that the documents do not say.

- Documents come as pasted texts or as absolute paths of text files and folders (`txt`, `md`, `csv`, `json`, `html` and similar). Limits: 300 documents, 400 000 characters each, 8 million in all, 500 cases.
- A reference passage is best given as its text (one or two sentences copied from the document). A retrieved passage counts as that reference when it contains the text, or holds at least 70 % of its content words (`match_threshold`), or shares its id; a reference with only a document name matches any passage of that document. A reference shorter than five content words must be contained whole. This is what lets one case set serve pipelines that cut their documents in different sizes.
- `require` says whether a case needs every reference passage (`all`, the default) or any of them.
- Cases are `proposed`, `accepted` or `rejected`. Only accepted cases are evaluated, unless a run asks for the proposed ones too.

### Cases proposed by a model

`rag_cases_generate` picks fragments spread over the documents and asks a model, for each, for a question a person could ask without having seen the fragment, the short answer and the sentence that supports it copied word for word. The quote is looked for in the fragment whatever the accents, the case and the line breaks, and the fragment's own text becomes the reference passage; when the model paraphrased, the closest sentence is used, and when nothing is close, the whole fragment. Questions the set already has are not proposed twice. With `unanswerable`, the model also writes questions the fragment does not answer; nothing checks that another document does not answer them. Generated cases are `proposed`: a person accepts, edits or rejects them on the page or with `rag_cases_review` and `rag_case_save`.

## What is measured

For every case the pipeline returns a trace: the passages retrieved in the retriever's order, the order after the reranker when there is one, the passages handed to the generator (the context) and the answer.

| Piece | Measures |
|---|---|
| Retriever | recall@k, nDCG@k (binary gains; a repeated passage adds nothing) for the cut-offs in `ks` (default 1, 3, 5, 10), MRR, the share of cases with some reference passage |
| Order and cut (reranker) | MRR, nDCG and recall at the size of the context, before and after the reranker, with the difference; cases the reranker rescued (outside the cut before, inside after) and cases it demoted |
| Generator | faithfulness to the passages it received, coverage of the expected answer, and abstention: how often it declines when the passage did not reach it, when it did, and on the cases with no answer; how often it is right when it has the passage |

A case passes when the context holds what the case needs and the answer is right: it does not decline, its faithfulness is at least `faithful_min` and its coverage at least `coverage_min` (0.6 both). A case with no answer passes when the generator declines. The pass rate carries a 95 % Wilson interval.

### How the answer is judged

- `logit`: the logit judge. Three one-token questions per case (faithfulness 0 to 9, coverage 0 to 9, answers or declines) answered from the next-token probabilities of a server that returns logprobs; a score is the expected value over the scale.
- `model`: one generated verdict in JSON from the judge model, for servers without logprobs.
- `lexical`: no model. Word overlap between the answer, the passages and the expected answer, and a list of phrases for abstention. Repeatable, blind to paraphrase.
- `auto` (default) uses the first that works in that order and each case records which one judged it. The judge is the `judge` argument or the setting `judge.contestant`; without one, `auto` is `lexical`. A run says when the model that wrote the answers also judged them.

## Attribution

The first piece that loses what the case needs is the cause.

| Piece | Cause | Meaning |
|---|---|---|
| `retriever` | `not_retrieved` | the reference passage is nowhere in what was retrieved |
| `retriever` | `cut_off` | it was retrieved below the cut and the pipeline has no reranker: the order is the retriever's own |
| `ranking` | `cut_off` | it was retrieved below the cut and the reranker did not lift it |
| `ranking` | `demoted` | it was inside the cut in the retriever's order and the reranker (or the pipeline's own filter) pushed it out |
| `generator` | `abstained` | it had the passage and said it did not know |
| `generator` | `invented` | it had the passage and its answer is not supported by the passages |
| `generator` | `incomplete` | it had the passage, stayed faithful, and did not give the expected answer |
| `generator` | `did_not_abstain` | it answered a question that has no answer |
| `generator` | `no_answer` | an empty reply |
| `unknown` | `no_trace`, `error` | the pipeline reported no passages, or the request failed |

`summary.attribution` gives the failures by piece with the share of each (`share.retriever` = 0.75 reads "75 % of the failures are the retriever's"), by cause, and the main piece. When the passage never reached the generator, what it did anyway is kept as a note: it declined as it should, it was right without the passage, or it invented.

## Pipelines

- `builtin`: this bench's own plain pipeline over the documents of the case set. `chunk_chars` and `chunk_overlap` cut the documents; the retriever is `bm25` (`depth` passages, default 10); the reranker is `none`, `overlap` (shared content words with the question) or `model` (the model scores each passage, by logits when the server allows); `top_k` passages go to the generator (default 3); the generator is `model` (a running server model answers from the passages and must decline when they do not say) or `extractive` (no model: the closest sentence). Broken pieces to check the attribution: retriever `random` or `none`, reranker `reverse` or `random`, generator `no_context` (the model never sees the passages).
- `faustus`: the document RAG of a running Faustus through its routes `POST /api/rag/answer` and `POST /api/rag/search`: what its chat retrieves for the question, which results pass its similarity threshold (the context), and what its default model answers with them. `url` defaults to the setting `faustus.url`; `source` is `documents` or `brain` (its notes search); `answer: model` writes the answer here with the chosen model from Faustus's passages. The documents must be indexed in that Faustus; the bench does not upload them.
- `http`: any other pipeline behind one request. `url`, `method`, `headers`, `body` (a JSON template where `{question}`, `{k}` and `{top_k}` are replaced), `passages_path` and `answer_path` (dotted paths into the reply, such as `data.hits`), optionally `ranked_path` and `context_path`, and `text_field`, `id_field`, `doc_field`, `score_field` for rows with other names. Without paths the usual names are tried. With `model`, a pipeline that only retrieves gets its answer written here. A pipeline that returns only an answer can be scored, but its failures cannot be split by piece.

`rag_probe` asks a pipeline one question and shows the trace without storing anything.

## Runs and comparison

`rag_run_start` runs in the background (`parallel` cases at a time, default 4); `rag_run_status` follows it, `rag_run_result` gives the summary by piece, the attribution and a row per case, and with `case` the whole trace of one case with the reference passages marked. A pipeline that fails five times before any case worked ends the run as failed with that error. Runs that were working when the app stopped are closed as failed.

`rag_compare` takes a baseline run and another over the same set (two fragment sizes, with and without reranker, two models): cases improved and regressed with an exact sign test, the change of each piece's measures, the failures by piece in each, where they moved, and what differs between the two pipeline descriptions. Runs of different sets are aligned by question.

## Tools and routes

Tools: `rag_sets`, `rag_set_save`, `rag_set_get`, `rag_set_delete`, `rag_cases_generate`, `rag_case_save`, `rag_cases_review`, `rag_case_remove`, `rag_probe`, `rag_run_start`, `rag_run_status`, `rag_run_result`, `rag_run_list`, `rag_run_cancel`, `rag_run_delete`, `rag_compare`. Deletes need `confirm=true`.

REST (under `/evaluation`): `GET|POST /api/rag/sets`, `GET|DELETE /api/rag/sets/{id}`, `POST /api/rag/sets/{id}/generate`, `POST /api/rag/sets/{id}/cases`, `POST /api/rag/sets/{id}/review`, `DELETE /api/rag/cases/{id}`, `POST /api/rag/probe`, `GET|POST /api/rag/runs`, `GET|DELETE /api/rag/runs/{id}`, `GET /api/rag/runs/{id}/status`, `POST /api/rag/runs/{id}/cancel`, `GET /api/rag/compare?a=&b=`.

## What it cannot do

- It loads no model: the generator, the judge and the case writer are servers that are already running.
- The built-in retriever is lexical (BM25). A vector retriever is measured through the `faustus` or `http` pipeline that has one.
- It reads text files only; PDF and office documents are not read as corpus.
- The match between a retrieved passage and a reference is by words, not by meaning: a paraphrased copy of the passage in another document does not count.
- The scores of the generator are a judgement of the judge model (or word overlap), not a measurement; with few cases the rates are weak and the comparison says so.
- A run does not survive a restart of the app: it is closed as failed and started again.
