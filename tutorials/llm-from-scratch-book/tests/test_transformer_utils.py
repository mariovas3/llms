from src.transformer_utils import utils


def test_pos_embeds(batch_3_5_7):
    pos_embed = utils.PosEmbed(batch_3_5_7.size(-2) + 5, batch_3_5_7.size(-1))
    out = pos_embed(batch_3_5_7)
    assert out.shape == batch_3_5_7.shape
