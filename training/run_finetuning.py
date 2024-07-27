from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from src.data.lit_data import LitInstructions
from src.model.lit_gpt import LitGPT


class MyLitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "trainer.accelerator",
            "data.on_gpu",
            compute_fn=lambda x: x == "gpu",
            apply_on="parse",
        )
        # make ModelCheckpoint callback configurable;
        parser.add_lightning_class_args(ModelCheckpoint, "my_model_checkpoint")
        parser.set_defaults(
            {
                "my_model_checkpoint.monitor": "validation/loss",
                "my_model_checkpoint.mode": "min",
                "my_model_checkpoint.every_n_epochs": 50,
            }
        )


def main():
    cli = MyLitCLI(
        model_class=LitGPT,
        datamodule_class=LitInstructions,
        seed_everything_default=0,
    )


if __name__ == "__main__":
    main()
