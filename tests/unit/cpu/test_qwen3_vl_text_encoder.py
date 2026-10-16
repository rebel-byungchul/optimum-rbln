from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from optimum.rbln.transformers.models.qwen3_vl.configuration_qwen3_vl import RBLNQwen3VLTextEncoderConfig
from optimum.rbln.transformers.models.qwen3_vl.modeling_qwen3_vl import RBLNQwen3VLTextEncoder


NUM_LAYERS = 4


def _tiny_qwen3_vl() -> Qwen3VLForConditionalGeneration:
    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        },
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 32},
    )
    config._attn_implementation = "sdpa"
    return Qwen3VLForConditionalGeneration(config).eval()


def _encoder_running_on_cpu(model, hidden_states_layers, max_seq_len) -> RBLNQwen3VLTextEncoder:
    """An RBLNQwen3VLTextEncoder whose compiled graph is replaced by the wrapper it would be compiled from."""
    rbln_config = RBLNQwen3VLTextEncoderConfig(max_seq_len=max_seq_len, hidden_states_layers=hidden_states_layers)
    encoder = RBLNQwen3VLTextEncoder.__new__(RBLNQwen3VLTextEncoder)
    encoder.rbln_config = rbln_config
    encoder.config = SimpleNamespace(text_config=model.config.text_config, return_dict=True)
    encoder.rotary_emb = Qwen3VLTextRotaryEmbedding(model.config.text_config)
    encoder.model = [RBLNQwen3VLTextEncoder._wrap_model_if_needed(model, rbln_config)]
    return encoder


@pytest.mark.parametrize("hidden_states_layers", [[1, 3], [0, 2, NUM_LAYERS]])
def test_hidden_states_match_the_original_with_right_padding(hidden_states_layers):
    """FLUX.3 encodes a right-padded prompt with explicit position ids and reads the padding positions too."""
    model = _tiny_qwen3_vl()
    seq_len, valid_len, max_seq_len = 10, 7, 16
    input_ids = torch.randint(1, 64, (1, seq_len))
    attention_mask = (torch.arange(seq_len) < valid_len).long()[None]
    position_ids = torch.arange(seq_len)[None].masked_fill(attention_mask == 0, 1)

    with torch.no_grad():
        expected = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_hidden_states=True
        ).hidden_states
        actual = _encoder_running_on_cpu(model, hidden_states_layers, max_seq_len)(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_hidden_states=True
        ).hidden_states

    assert len(actual) == NUM_LAYERS + 1
    for layer in range(NUM_LAYERS + 1):
        if layer in hidden_states_layers:
            torch.testing.assert_close(actual[layer], expected[layer], rtol=1e-5, atol=1e-5)
        else:
            assert actual[layer] is None


def test_compiles_only_the_layers_up_to_the_last_requested():
    model = _tiny_qwen3_vl()
    rbln_config = RBLNQwen3VLTextEncoderConfig(hidden_states_layers=[2, 1])
    RBLNQwen3VLTextEncoder._update_rbln_config(None, model, model.config, rbln_config)

    assert rbln_config.hidden_states_layers == [1, 2]
    assert len(RBLNQwen3VLTextEncoder._wrap_model_if_needed(model, rbln_config).layers) == 2
    names = [name for name, _, _ in rbln_config.compile_cfgs[0].input_info]
    assert names == ["input_ids", "attention_mask", "position_cos", "position_sin"]


def test_rejects_prompts_longer_than_compiled():
    encoder = _encoder_running_on_cpu(_tiny_qwen3_vl(), [1], max_seq_len=4)
    with pytest.raises(ValueError, match="max_seq_len"):
        encoder(input_ids=torch.ones(1, 5, dtype=torch.long))
