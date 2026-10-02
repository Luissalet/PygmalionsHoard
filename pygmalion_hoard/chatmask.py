"""Which tokens of a conversation the model is trained on. Pure Python (standard library only): the trainer worker loads this file
by path and the tests run it with a fake chat template, so none of it needs torch or a tokenizer.

The rule: only the assistant's turns (and the end-of-turn marker that closes each one) carry a label; the system and user turns
and the template's own scaffolding are masked with ``IGNORE_INDEX``.

Two ways of finding the assistant's characters in the rendered conversation:

* ``prefix``: render the conversation up to turn *i* with the generation prompt, and up to and including turn *i*; the text
  between the two is what the model must produce. Exact, and works for every template whose earlier turns do not change when a
  later one is added.
* ``content``: for templates that rewrite earlier turns (for example by dropping reasoning blocks), look the assistant's text up
  in the full rendering and extend each match with the end-of-turn marker learned from a probe conversation.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

IGNORE_INDEX = -100

Render = Callable[[list[dict[str, Any]], bool], str]
Encode = Callable[[str], tuple[list[int], Optional[list[tuple[int, int]]]]]


def probe_end_of_turn(render: Render) -> str:
    """The text a template appends after an assistant message (``<|im_end|>\\n`` and the like)."""
    mark = "PYGMALIONPROBE"
    try:
        text = render([{"role": "user", "content": "q"}, {"role": "assistant", "content": mark}], False)
    except Exception:  # noqa: BLE001 — a template that rejects the probe has no usable marker
        return ""
    at = text.rfind(mark)
    return text[at + len(mark):] if at != -1 else ""


def spans_by_prefix(messages: Sequence[dict[str, Any]], render: Render) -> Optional[tuple[str, list[tuple[int, int]]]]:
    """Character spans of the assistant turns inside the full rendering, or None when the template is not prefix-stable."""
    full = render(list(messages), False)
    spans: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        start_text = render(list(messages[:index]), True)
        end_text = render(list(messages[: index + 1]), False)
        if not (end_text.startswith(start_text) and full.startswith(end_text)):
            return None
        if len(end_text) > len(start_text):
            spans.append((len(start_text), len(end_text)))
    return full, spans


def spans_by_content(messages: Sequence[dict[str, Any]], render: Render) -> tuple[str, list[tuple[int, int]]]:
    """Fallback: find each assistant message in the full rendering, in order, and add the end-of-turn marker."""
    full = render(list(messages), False)
    eot = probe_end_of_turn(render)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        at = full.find(content, cursor)
        if at == -1:
            continue
        end = at + len(content)
        if eot and full.startswith(eot, end):
            end += len(eot)
        spans.append((at, end))
        cursor = end
    return full, spans


def assistant_spans(messages: Sequence[dict[str, Any]], render: Render) -> tuple[str, list[tuple[int, int]], str]:
    """``(full text, spans, mode)`` with mode ``prefix`` or ``content``."""
    try:
        found = spans_by_prefix(messages, render)
    except Exception:  # noqa: BLE001 — some templates raise on a conversation that ends with the user
        found = None
    if found is not None:
        return found[0], found[1], "prefix"
    full, spans = spans_by_content(messages, render)
    return full, spans, "content"


def label_tokens(offsets: Sequence[tuple[int, int]], spans: Sequence[tuple[int, int]]) -> list[bool]:
    """True for every token whose first character lies inside an assistant span."""
    flags: list[bool] = []
    index = 0
    ordered = sorted(spans)
    for start, _end in offsets:
        while index < len(ordered) and ordered[index][1] <= start:
            index += 1
        flags.append(index < len(ordered) and ordered[index][0] <= start < ordered[index][1])
    return flags


def build_chat_example(messages: Sequence[dict[str, Any]], render: Render, encode: Encode, max_len: Optional[int] = None,
                       train_on: str = "assistant") -> dict[str, Any]:
    """Token ids and labels of one conversation.

    ``encode(text)`` returns ``(ids, offsets)``; ``offsets`` may be None for tokenizers without character offsets, in which case
    the text is encoded piece by piece. ``train_on`` is ``assistant`` (every assistant turn) or ``last`` (only the final one).
    """
    full, spans, mode = assistant_spans(messages, render)
    if train_on == "last" and spans:
        spans = spans[-1:]
    ids, offsets = encode(full)
    if offsets is not None and len(offsets) == len(ids):
        flags = label_tokens(offsets, spans)
        labels = [token if flag else IGNORE_INDEX for token, flag in zip(ids, flags)]
    else:
        ids, labels = [], []
        cursor = 0
        for start, end in sorted(spans):
            if start > cursor:
                piece, _ = encode(full[cursor:start])
                ids += piece
                labels += [IGNORE_INDEX] * len(piece)
            piece, _ = encode(full[start:end])
            ids += piece
            labels += piece
            cursor = end
        if cursor < len(full):
            piece, _ = encode(full[cursor:])
            ids += piece
            labels += [IGNORE_INDEX] * len(piece)
    if max_len is not None and len(ids) > max_len:
        ids, labels = ids[:max_len], labels[:max_len]
    trainable = sum(1 for x in labels if x != IGNORE_INDEX)
    return {"input_ids": ids, "labels": labels, "trainable": trainable, "mode": mode}


def build_completion_example(prompt: str, response: str, encode_plain: Callable[[str], list[int]], eos_id: Optional[int],
                             max_len: Optional[int] = None) -> dict[str, Any]:
    """A prompt/response pair without a chat template: loss on the response (and the end-of-sequence token) only."""
    prompt_ids = encode_plain(prompt)
    response_ids = encode_plain(response)
    if eos_id is not None:
        response_ids = response_ids + [eos_id]
    ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    if max_len is not None and len(ids) > max_len:
        ids, labels = ids[:max_len], labels[:max_len]
    return {"input_ids": ids, "labels": labels, "trainable": sum(1 for x in labels if x != IGNORE_INDEX), "mode": "completion"}


def pack_sequences(sequences: Sequence[Sequence[int]], seq_len: int, eos_id: Optional[int]) -> list[list[int]]:
    """Raw text for continued pretraining: join the documents with the end-of-sequence token and cut chunks of ``seq_len``.
    The remainder shorter than half a chunk is dropped; a longer one is kept as a shorter last chunk."""
    stream: list[int] = []
    for sequence in sequences:
        stream.extend(sequence)
        if eos_id is not None:
            stream.append(eos_id)
    chunks = [stream[i:i + seq_len] for i in range(0, len(stream), seq_len)]
    if chunks and len(chunks[-1]) < max(2, seq_len // 2):
        chunks.pop()
    return [c for c in chunks if len(c) >= 2]


def instruction_to_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    """``{"prompt", "response"}`` (and an optional ``system``) as chat messages."""
    messages: list[dict[str, Any]] = []
    if record.get("system"):
        messages.append({"role": "system", "content": str(record["system"])})
    messages.append({"role": "user", "content": str(record.get("prompt", ""))})
    messages.append({"role": "assistant", "content": str(record.get("response", ""))})
    return messages
