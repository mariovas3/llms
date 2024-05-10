from src.transformer_utils import utils


def test_pos_embeds(dummy_embedded_batch):
    pos_embed = utils.PosEmbed(dummy_embedded_batch.size(-2) + 5, dummy_embedded_batch.size(-1))
    out = pos_embed(dummy_embedded_batch)
    assert out.shape == dummy_embedded_batch.shape