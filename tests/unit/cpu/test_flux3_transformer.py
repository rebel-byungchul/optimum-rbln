from types import SimpleNamespace

import pytest
import torch
from diffusers.models.embeddings import Timesteps
from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel

from optimum.rbln.diffusers.configurations import RBLNFlux3Transformer2DModelConfig
from optimum.rbln.diffusers.models.transformers.transformer_flux3 import (
    TEXT_PAD_MULTIPLE,
    Flux3Transformer2DModelWrapper,
    RBLNFlux3Transformer2DModel,
    get_image_ids,
    get_text_ids,
)


SAMPLE_SIZE = (4, 6)
IMAGE_ROPE_SCALE = 0.5


def _tiny_transformer(flow_map: bool) -> Flux3Transformer2DModel:
    torch.manual_seed(0)
    model = Flux3Transformer2DModel(
        in_channels=8,
        cond_in_channels=8,
        num_layers=1,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=24,
        timestep_channels=16,
        axes_dims_rope=(4, 4, 4, 4),
        flow_map=flow_map,
    )
    # Freshly initialized modulations are zero; randomize every weight so each path contributes.
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.1)
    return model.eval()


def _rbln_model_running_on_cpu(model: Flux3Transformer2DModel, max_seq_len: int) -> RBLNFlux3Transformer2DModel:
    """An RBLNFlux3Transformer2DModel whose compiled graph is replaced by the wrapper it would be compiled from."""
    rbln_config = RBLNFlux3Transformer2DModelConfig(
        sample_size=SAMPLE_SIZE, max_seq_len=max_seq_len, image_rope_scale=IMAGE_ROPE_SCALE
    )
    rbln_model = RBLNFlux3Transformer2DModel.__new__(RBLNFlux3Transformer2DModel)
    rbln_model.rbln_config = rbln_config
    rbln_model.config = SimpleNamespace(flow_map=model.config.flow_map)
    rbln_model.time_proj = Timesteps(
        num_channels=model.config.timestep_channels, flip_sin_to_cos=True, downscale_freq_shift=0
    )
    rbln_model.image_ids = get_image_ids(SAMPLE_SIZE, IMAGE_ROPE_SCALE)
    rbln_model.text_ids = get_text_ids(max_seq_len)
    rbln_model.model = [RBLNFlux3Transformer2DModel._wrap_model_if_needed(model, rbln_config)]
    return rbln_model


@pytest.mark.parametrize("flow_map", [False, True])
def test_padded_and_masked_text_matches_the_original(flow_map):
    model = _tiny_transformer(flow_map)
    text_len, max_seq_len = 2 * TEXT_PAD_MULTIPLE, 4 * TEXT_PAD_MULTIPLE
    image_ids = get_image_ids(SAMPLE_SIZE, IMAGE_ROPE_SCALE)
    text_ids = get_text_ids(text_len)
    inputs = {
        "hidden_states": torch.randn(1, SAMPLE_SIZE[0] * SAMPLE_SIZE[1], 8),
        "encoder_hidden_states": torch.randn(1, text_len, 24),
        "timestep": torch.tensor([0.73]),
        "img_ids": image_ids,
        "txt_ids": text_ids,
        "target_timestep": torch.tensor([0.61]) if flow_map else None,
    }

    with torch.no_grad():
        expected = model(**inputs, return_dict=False)[0]
        actual = _rbln_model_running_on_cpu(model, max_seq_len)(**inputs, return_dict=False)[0]

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_rejects_positions_the_graph_was_not_compiled_for():
    model = _rbln_model_running_on_cpu(_tiny_transformer(flow_map=False), TEXT_PAD_MULTIPLE)
    shifted_ids = get_image_ids(SAMPLE_SIZE, IMAGE_ROPE_SCALE) + 1
    with pytest.raises(ValueError, match="img_ids"):
        model(
            hidden_states=torch.randn(1, SAMPLE_SIZE[0] * SAMPLE_SIZE[1], 8),
            encoder_hidden_states=torch.randn(1, TEXT_PAD_MULTIPLE, 24),
            timestep=torch.tensor([0.5]),
            img_ids=shifted_ids,
            txt_ids=get_text_ids(TEXT_PAD_MULTIPLE),
        )


def test_rejects_text_longer_than_compiled():
    model = _rbln_model_running_on_cpu(_tiny_transformer(flow_map=False), TEXT_PAD_MULTIPLE)
    text_len = 2 * TEXT_PAD_MULTIPLE
    with pytest.raises(ValueError, match="max_seq_len"):
        model(
            hidden_states=torch.randn(1, SAMPLE_SIZE[0] * SAMPLE_SIZE[1], 8),
            encoder_hidden_states=torch.randn(1, text_len, 24),
            timestep=torch.tensor([0.5]),
            img_ids=get_image_ids(SAMPLE_SIZE, IMAGE_ROPE_SCALE),
            txt_ids=get_text_ids(text_len),
        )


def test_wrapper_graph_inputs_match_the_compile_config():
    model = _tiny_transformer(flow_map=True)
    rbln_config = RBLNFlux3Transformer2DModelConfig(sample_size=SAMPLE_SIZE, max_seq_len=100)
    RBLNFlux3Transformer2DModel._update_rbln_config(None, model, model.config, rbln_config)

    assert rbln_config.max_seq_len == 2 * TEXT_PAD_MULTIPLE  # rounded up to the pipeline's pad multiple
    names = [name for name, _, _ in rbln_config.compile_cfgs[0].input_info]
    assert names == [
        "hidden_states",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "timestep_proj",
        "target_timestep_proj",
    ]
    wrapper = Flux3Transformer2DModelWrapper(
        model, get_image_ids(SAMPLE_SIZE, rbln_config.image_rope_scale), get_text_ids(rbln_config.max_seq_len)
    )
    dummy = [
        torch.zeros(shape, dtype=getattr(torch, dtype)) for _, shape, dtype in rbln_config.compile_cfgs[0].input_info
    ]
    with torch.no_grad():
        (output,) = wrapper(*dummy)
    assert output.shape == (1, SAMPLE_SIZE[0] * SAMPLE_SIZE[1], 8)
