"""Which tokens carry a label, with a fake chat template and a character-level tokenizer (no torch)."""

from pygmalion_hoard import chatmask as C

IGN = C.IGNORE_INDEX


def render(messages, add_generation_prompt):
    out = ""
    for m in messages:
        out += f"<|{m['role']}|>{m['content']}<|end|>\n"
    if add_generation_prompt:
        out += "<|assistant|>"
    return out


def render_dropping_history(messages, add_generation_prompt):
    """A template that rewrites earlier assistant turns (like one that strips reasoning)."""
    last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    out = ""
    for i, m in enumerate(messages):
        body = m["content"] if (m["role"] != "assistant" or i == last) else m["content"].upper()
        out += f"<|{m['role']}|>{body}<|end|>\n"
    if add_generation_prompt:
        out += "<|assistant|>"
    return out


def encode(text):
    ids = [ord(c) for c in text]
    return ids, [(i, i + 1) for i in range(len(text))]


def encode_no_offsets(text):
    return [ord(c) for c in text], None


MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hola"}, {"role": "assistant", "content": "buenas"}]


def text_of(ex):
    return "".join(chr(t) for t, l in zip(ex["input_ids"], ex["labels"]) if l != IGN)


def test_only_the_assistant_turn_and_its_end_marker_are_labelled():
    ex = C.build_chat_example(MSGS, render, encode)
    assert text_of(ex) == "buenas<|end|>\n" and ex["mode"] == "prefix"
    assert ex["trainable"] == len("buenas<|end|>\n")


def test_labels_equal_the_input_ids_where_set():
    ex = C.build_chat_example(MSGS, render, encode)
    assert all(l == t for t, l in zip(ex["input_ids"], ex["labels"]) if l != IGN)
    assert len(ex["input_ids"]) == len(ex["labels"])


def test_the_generation_prompt_is_not_trained_on():
    ex = C.build_chat_example(MSGS, render, encode)
    assert "<|assistant|>" not in text_of(ex)


def test_two_assistant_turns_are_both_labelled_or_only_the_last():
    msgs = MSGS + [{"role": "user", "content": "otra"}, {"role": "assistant", "content": "vale"}]
    both = C.build_chat_example(msgs, render, encode)
    last = C.build_chat_example(msgs, render, encode, train_on="last")
    assert text_of(both) == "buenas<|end|>\nvale<|end|>\n" and text_of(last) == "vale<|end|>\n"


def test_an_unstable_template_falls_back_to_content_matching():
    msgs = MSGS + [{"role": "user", "content": "otra"}, {"role": "assistant", "content": "vale"}]
    ex = C.build_chat_example(msgs, render_dropping_history, encode)
    assert ex["mode"] == "content" and "vale<|end|>" in text_of(ex)


def test_tokenizers_without_offsets_are_encoded_piece_by_piece():
    ex = C.build_chat_example(MSGS, render, encode_no_offsets)
    assert text_of(ex) == "buenas<|end|>\n"


def test_max_len_truncates_both_arrays():
    ex = C.build_chat_example(MSGS, render, encode, max_len=10)
    assert len(ex["input_ids"]) == 10 == len(ex["labels"])


def test_probe_end_of_turn():
    assert C.probe_end_of_turn(render) == "<|end|>\n"
    assert C.probe_end_of_turn(lambda m, g: (_ for _ in ()).throw(ValueError("no"))) == ""


def test_label_tokens_uses_the_first_character_of_each_token():
    flags = C.label_tokens([(0, 3), (3, 6), (6, 9)], [(3, 6)])
    assert flags == [False, True, False]


def test_completion_example_masks_the_prompt_and_adds_eos():
    ex = C.build_completion_example("pregunta ", "respuesta", lambda s: [ord(c) for c in s], 0)
    assert ex["labels"][: len("pregunta ")] == [IGN] * len("pregunta ")
    assert ex["labels"][-1] == 0 and ex["trainable"] == len("respuesta") + 1 and ex["mode"] == "completion"


def test_completion_example_without_eos_and_with_limit():
    ex = C.build_completion_example("ab", "cd", lambda s: [ord(c) for c in s], None, max_len=3)
    assert len(ex["input_ids"]) == 3


def test_pack_sequences_cuts_chunks_and_drops_a_short_remainder():
    chunks = C.pack_sequences([[1] * 10, [2] * 10], 8, 99)
    assert all(len(c) <= 8 for c in chunks) and sum(len(c) for c in chunks) <= 22
    assert chunks[0] == [1] * 8
    short = C.pack_sequences([[1] * 8, [2]], 8, None)
    assert short == [[1] * 8]


def test_instruction_to_messages():
    m = C.instruction_to_messages({"prompt": "p", "response": "r", "system": "s"})
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    assert [x["role"] for x in C.instruction_to_messages({"prompt": "p", "response": "r"})] == ["user", "assistant"]
