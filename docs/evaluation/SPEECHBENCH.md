# Speech bench: how well an engine turns speech into text

Engines transcribe a corpus whose references were written by a person, and every hypothesis is scored and timed. The point
is to decide which recogniser to use for dictation, subtitles or the voice mode with numbers measured on this computer, not
with the figures each project publishes.

## The corpus

A folder (`speechbench_corpus` checks it):

- **pairs**: every audio file (`.wav .mp3 .flac .m4a .ogg .opus .aac .webm .mp4`, subfolders too) with its reference next to
  it under the same name (`clip01.wav` + `clip01.txt`, UTF-8); or
- **`corpus.jsonl`** in the folder: one `{"audio": "relative/or/absolute/path", "text": "reference", "id": "optional"}` per line.

Files without a reference, or with an empty one, are listed as problems and never scored. Durations come from the WAV
header or ffprobe; a file whose length is unknown is scored but has no real-time factor. References must be independent:
a reference written by a recogniser (or the prompt a voice was synthesised from) makes WER a measure of agreement, not of
accuracy, and should be reported as a smoke test.

## Engines

- `faster-whisper:<size>` (`tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`): `galton_hoard/speechbench/whisper_worker.py`
  runs in a Python that has faster-whisper - the one set with `speechbench_engines(whisper_python=...)`, `GALTON_WHISPER_PYTHON`,
  this app's own Python, or the venv of another app of the family as the hub lists it. One process and one warm model for the
  whole corpus; models from the family's Whisper folder (`~/.hoard/models/whisper`); CPU with int8 by default (`device`,
  `compute`), at below-normal priority, so the GPUs stay free.
- `cmd:<name>`: a command saved with `speechbench_engines(name, command=[...])`, run once per file without a shell; `{audio}` is
  the file and `{language}` the language code. It prints plain text, or JSON `{"text": ..., "first_text_s": ...}` (also as its
  last line) when it streams. Its time includes loading the model, since every file is a new process. This is how a streaming
  model such as Confucius4-R2T2 (its CPU llama.cpp build) is compared with Whisper.

## What is measured

Before comparing, both texts are lower-cased, punctuation is removed and spaces collapsed; `accents=false` also folds
accents (`canción` = `cancion`). Numbers are compared as written: «18» against «dieciocho» is an error.

- **WER / CER of the corpus**: total word (character) edits over total reference words (characters), with the substitutions,
  deletions and insertions behind it. Not the mean of per-file rates, so a long file weighs what it says.
- **Load**: seconds to load the model (warm engines only).
- **First text**: seconds from the start of decoding a file to the first segment the decoder yields (Whisper decodes in
  30 s windows, so this is its first window) or the `first_text_s` a command engine reports; the median per engine.
- **Decode and RTF**: wall time per file, and total decode time over total audio time (under 1 is faster than real time).
- **Failures**: files an engine could not transcribe, with the reason.

Runs live in `<data>/speechbench/runs/<id>.json` and are rewritten as files finish, so the screen and
`speechbench_result` show a run in progress. One file after another, one engine after another.

## Surfaces

- MCP / `POST /api/agent/call` / `POST /api/ui/call`: `speechbench_engines`, `speechbench_corpus`, `speechbench_start`
  (`dry_run` shows the plan), `speechbench_result` (`items=true` for every file), `speechbench_list`, `speechbench_cancel`.
- Screen **Voz** of the evaluation: corpus folder and check, engines, language, accent folding, start; the command-engine and
  Whisper-Python form; runs with the per-engine table and every file's reference, transcript, WER and RTF.
