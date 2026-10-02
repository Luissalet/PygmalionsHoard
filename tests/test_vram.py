"""The VRAM and time estimator."""

import pytest

from pygmalion_hoard import vram as V

CFG = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336,
       "vocab_size": 128256}


def test_param_estimate_is_near_eight_billion():
    n = V.estimate_params(CFG)
    assert 7.5e9 < n < 8.6e9


def test_tied_embeddings_save_parameters():
    assert V.estimate_params({**CFG, "tie_word_embeddings": True}) < V.estimate_params(CFG)


def test_multimodal_config_uses_the_text_section():
    wrapped = {"vision_config": {"hidden_size": 1}, "text_config": CFG}
    assert V.estimate_params(wrapped) == V.estimate_params(CFG)


def test_moe_counts_every_expert_but_adapts_none():
    moe = {**CFG, "num_experts": 8, "moe_intermediate_size": 2048}
    assert V.estimate_params(moe) != V.estimate_params(CFG)
    assert V.lora_params(moe, 16) < V.lora_params(CFG, 16)


def test_lora_params_scale_with_rank():
    assert V.lora_params(CFG, 32) == 2 * V.lora_params(CFG, 16)


def test_empty_config_gives_zero():
    assert V.estimate_params({}) == 0 and V.lora_params({}, 8) == 0


def test_qlora_needs_much_less_than_lora():
    q = V.estimate_training_memory(CFG, method="qlora")["total_mb"]
    l = V.estimate_training_memory(CFG, method="lora")["total_mb"]
    assert q < l * 0.6     # the bf16 embeddings and head keep QLoRA a little above half of LoRA


def test_memory_grows_with_sequence_length_and_batch():
    base = V.estimate_training_memory(CFG, seq_len=1024)["total_mb"]
    assert V.estimate_training_memory(CFG, seq_len=4096)["total_mb"] > base
    assert V.estimate_training_memory(CFG, seq_len=1024, batch=4)["total_mb"] > base


def test_the_parts_add_up_to_the_total_within_rounding():
    e = V.estimate_training_memory(CFG)
    parts = e["weights_mb"] + e["lora_mb"] + e["checkpoints_mb"] + e["working_mb"] + e["logits_mb"] + e["overhead_mb"]
    assert abs(parts - e["total_mb"]) <= 6
    assert "weights" in e["formula"] and "overhead" in e["formula"]


def test_unknown_method_is_treated_as_qlora():
    assert V.estimate_training_memory(CFG, method="whatever")["method"] == "qlora"


GPUS = [{"index": 2, "name": "A", "total_mb": 16000, "free_mb": 15000}, {"index": 3, "name": "B", "total_mb": 16000, "free_mb": 4000}]


def test_fit_on_gpus_reports_free_and_total():
    out = V.fit_on_gpus(10000, GPUS)
    assert out[0]["fits_free"] and not out[1]["fits_free"] and out[1]["fits_total"]


def test_pick_gpus_prefers_one_card_with_most_free_memory():
    got = V.pick_gpus(10000, GPUS)
    assert got["gpus"] == [2] and got["fits"] is True and got["waits"] is False


def test_pick_gpus_waits_when_free_memory_is_short():
    got = V.pick_gpus(15500, [{"index": 3, "total_mb": 16000, "free_mb": 2000}])
    assert got["gpus"] == [3] and got["waits"] is True


def test_pick_gpus_splits_over_several_cards():
    got = V.pick_gpus(24000, GPUS)
    assert got["gpus"] == [2, 3] and got["split"] is True


def test_pick_gpus_says_when_nothing_fits():
    got = V.pick_gpus(60000, GPUS)
    assert got["fits"] is False and got["gpus"] == [] and "60000" in got["reason"]


def test_pick_gpus_without_inventory():
    assert V.pick_gpus(1000, [])["fits"] is None


QWEN_06B = {"hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
            "intermediate_size": 3072, "vocab_size": 151936, "tie_word_embeddings": True}


def test_estimate_time_uses_the_card_speed():
    slow = V.estimate_time(8_000_000_000, 1_000_000, 1.0, "qlora", "unknown card")
    fast = V.estimate_time(8_000_000_000, 1_000_000, 1.0, "qlora", "RTX 5090")
    assert fast["seconds"] < slow["seconds"] and fast["assumed_tflops"] == 200.0 and fast["gpu_scale"] > 1 > slow["gpu_scale"]


def test_the_reference_constant_is_1600_tokens_per_second_per_billion_params_for_qlora_on_a_16gb_50_series_card():
    assert V.throughput(1_000_000_000, "qlora", "NVIDIA GeForce RTX 5060 Ti") == 1600.0
    assert V.throughput(2_000_000_000, "qlora", "NVIDIA GeForce RTX 5060 Ti") == 800.0
    assert V.THROUGHPUT_CONSTANT == 1600.0 and V.METHOD_SPEED == {"qlora": 1.0, "lora": 1.6}


def test_plain_lora_is_one_point_six_times_faster_than_qlora():
    q = V.throughput(4_000_000_000, "qlora", "RTX 5060 Ti")
    assert V.throughput(4_000_000_000, "lora", "RTX 5060 Ti") == pytest.approx(q * 1.6)
    assert V.estimate_time(4_000_000_000, 400_000, 1, "lora", "RTX 5060 Ti")["seconds"] < V.estimate_time(4_000_000_000, 400_000, 1, "qlora", "RTX 5060 Ti")["seconds"]


def test_a_tiny_model_is_capped_by_the_launch_bound_ceiling():
    assert V.throughput(10_000_000, "qlora", "RTX 5060 Ti") == V.MAX_TOKENS_PER_S
    assert V.throughput(0, "qlora") == 0.0


@pytest.mark.parametrize("epochs", [0.5, 1, 3])
def test_the_time_beyond_the_start_up_is_proportional_to_the_tokens_to_process(epochs):
    base = V.estimate_time(1_000_000_000, 100_000, 1.0, "lora")
    got = V.estimate_time(1_000_000_000, 100_000, epochs, "lora")
    assert got["tokens"] == int(100_000 * epochs)
    assert abs((got["seconds"] - got["startup_s"]) - (base["seconds"] - base["startup_s"]) * epochs) <= 2


def test_the_time_includes_the_start_up_and_the_per_step_overhead():
    none = V.estimate_time(1_000_000_000, 0, 1.0, "qlora", "RTX 5060 Ti", steps=0)
    assert none["seconds"] == none["startup_s"] == round(V.STARTUP_S + V.STARTUP_S_PER_B)
    with_steps = V.estimate_time(1_000_000_000, 0, 1.0, "qlora", "RTX 5060 Ti", steps=100)
    assert with_steps["seconds"] - none["seconds"] == pytest.approx(100 * V.STEP_OVERHEAD_S, abs=1)


def test_a_qlora_of_a_0_6b_model_on_140_short_records_for_3_epochs_is_minutes_not_seconds():
    """The defect: 9 s for 27 steps. 126 records x 3 epochs x ~500 tokens at seq 768 is about 190k tokens to process."""
    params = V.estimate_params(QWEN_06B)
    assert 0.5e9 < params < 0.8e9
    got = V.estimate_time(params, 140 * 450, 3.0, "qlora", "NVIDIA GeForce RTX 5060 Ti", steps=27)
    assert 2000 < got["tokens_per_s"] < 3500                       # about 1600 / 0.64
    assert 90 <= got["seconds"] <= 400
    assert got["tokens"] == 140 * 450 * 3 and got["source"] == "estimate"


def test_the_time_formula_shows_every_number_it_used():
    got = V.estimate_time(2_000_000_000, 300_000, 2.0, "qlora", "RTX 5060 Ti", steps=40)
    formula = str(got["formula"])
    for fragment in ("600000 tokens", "40 steps", "800 tokens/s", "1600 / 2.0B", "qlora"):
        assert fragment in formula
    assert got["formula"].key == "time_formula" and got["note"].key == "time_note"


def test_qlora_keeps_the_embeddings_and_head_in_bf16():
    """bitsandbytes leaves the embeddings and the output head unquantized: with a large vocabulary that is gigabytes the
    estimate (and so the GPU lease) must include."""
    big_vocab = {**CFG, "vocab_size": 248_320}
    e = V.estimate_training_memory(big_vocab, method="qlora")
    n, kept = e["params"], V.embedding_params(big_vocab)
    assert kept == 248_320 * 4096 * 2
    assert e["weights_mb"] == round(((n - kept) * 0.55 + kept * 2.0) / 1048576)
    tied = V.estimate_training_memory({**big_vocab, "tie_word_embeddings": True}, method="qlora")
    assert tied["weights_mb"] < e["weights_mb"] and "embeddings" in e["formula"]
