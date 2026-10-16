import pytest

from optimum.rbln import RBLNFlux3DistilledModularPipeline, RBLNFlux3DistilledModularPipelineConfig


def test_defaults_share_the_image_size_and_prompt_length():
    config = RBLNFlux3DistilledModularPipelineConfig()

    assert config.image_size == (1024, 1024)
    assert config.vae.sample_size == (1024, 1024)
    assert config.vae.uses_encoder is False
    assert config.transformer.sample_size == (64, 64)
    assert config.transformer.max_seq_len == config.text_encoder.max_seq_len == 1040
    assert config.text_encoder.hidden_states_layers == [4, 8, 12, 16, 20, 24, 28, 32]
    # The upsampler writes the caption: room for its system prompt and 3584 new tokens, flash attention.
    assert config.prompt_upsampler.max_seq_len == 8192
    assert config.prompt_upsampler.kvcache_partition_len == 4096
    assert config.prompt_upsampler.batch_size == 1


def test_size_and_prompt_length_reach_every_submodule():
    config = RBLNFlux3DistilledModularPipelineConfig(height=768, width=1360, max_seq_len=500, batch_size=2)

    assert config.vae.sample_size == (768, 1360)
    assert config.transformer.sample_size == (48, 85)
    # Rounded up to the pipeline's 80-token pad window, identically for both submodules.
    assert config.transformer.max_seq_len == config.text_encoder.max_seq_len == 560
    assert config.text_encoder.batch_size == config.transformer.batch_size == config.vae.batch_size == 2


@pytest.mark.parametrize("size", [{"height": 1024}, {"height": 1000, "width": 1024}])
def test_rejects_invalid_image_sizes(size):
    with pytest.raises(ValueError):
        RBLNFlux3DistilledModularPipelineConfig(**size)


def test_pipeline_resolves_its_config_class():
    assert RBLNFlux3DistilledModularPipeline.get_rbln_config_class() is RBLNFlux3DistilledModularPipelineConfig
