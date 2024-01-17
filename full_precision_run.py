from data import MyDataModule

import torch
from lightning.pytorch import LightningModule

from sr_experiments.network import PreResNet
from torch import nn
from torch.optim import SGD
from lightning.pytorch.callbacks import ModelCheckpoint

import torch.nn.functional as F
import argparse
import lightning as L
from utils import make_quant_func, Id


from lightning.pytorch.loggers import TensorBoardLogger


rounding_options_choices = ["nearest", "stochastic", ]

parser = argparse.ArgumentParser(description=' Example')
parser.add_argument('-b', "--batch-size", type=int, default=128, help='batch size')
parser.add_argument('-l', "--learning-rate", type=float, default=0.1, help='learning rate')
parser.add_argument('-m', "--momentum", type=float, default=0, help='momentum')
# parser.add_argument('-w', "--weight-bw", type=int, default=3, help='mantissa bit width for weight')
# parser.add_argument('-e', "--error-bw", type=int, default=3, help='mantissa bit width for error')
# parser.add_argument('-g', "--gradient-bw", type=int, default=3, help='mantissa bit width for gradient')
# parser.add_argument('-a', "--activation-bw", type=int, default=3, help='mantissa bit width for activation')
# parser.add_argument("--weight-round", default="nearest", choices=rounding_options_choices, help='weight rounding method')
# parser.add_argument("--error-round",  default="nearest",choices=rounding_options_choices, help='error rounding method')
# parser.add_argument("--gradient-round", default="nearest", choices=rounding_options_choices, help='gradient rounding method')
# parser.add_argument("--activation-round", default="nearest", choices=rounding_options_choices, help='activation rounding method')
# parser.add_argument("--checkpoint-path", default=None, help='checkpoint path')

device = "cuda" if torch.cuda.is_available() else "cpu"

args = parser.parse_args()



class LitClassifier(LightningModule):
    def _apply_model_weights(self, model, quant_func):
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data = quant_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data = quant_func(m.bias.data)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = PreResNet(lambda : Id())
        #self.automatic_optimization=False
        self.save_hyperparameters(ignore=["backbone"])

    def forward(self, x):
        # use forward for inference/predictions
        return self.backbone(x)

    def copy_model_weights_to_reference_weights(self):
        for (name, param), (ref_name, ref_param) in zip(
            self.backbone.named_parameters(), 
            self.reference_model.named_parameters()):
            ref_param.data = param.data.clone()




    def log_dict_hist(self, dict, prefix = ""):
        tensorboard = self.logger.experiment
        for name, tensor in dict.items():
            tensorboard.add_histogram(prefix + name, tensor, self.current_epoch)


    def opt_step(self, loss, opt):
        opt.step()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        # self.log("valid_loss", loss, on_step=True)
        self.log_dict({"valid_loss": loss, "valid_acc": acc, }, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log_dict({"test_loss": loss, "test_acc": acc,}, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = SGD(self.parameters(), 
                        lr=self.args.learning_rate, 
                        momentum=self.args.momentum)
        return optimizer




def make_version_name(args):
    return (
        f"w{args.weight_bw}{args.weight_round[0]}"
        f"e{args.error_bw}{args.error_round[0]}"
        f"g{args.gradient_bw}{args.gradient_round[0]}"
        f"a{args.activation_bw}{args.activation_round[0]}"
        f"lr{args.learning_rate}"
        f"b{args.batch_size}"
    )



def cli_main():
    args = parser.parse_args()
    model = LitClassifier(args)
    print(args)
    logger = TensorBoardLogger("mix_precision_vs_stochastic", name="full_precision_reference", )
    
    checkpoint_callback = ModelCheckpoint(every_n_epochs=5)
    trainer = L.Trainer(accelerator="gpu", max_epochs=100, logger=logger, devices=1, callbacks=[checkpoint_callback])
    datamodule = MyDataModule(args.batch_size)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    cli_main()
