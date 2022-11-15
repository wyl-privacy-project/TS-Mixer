from dataset import TSMixerDataModule
from model import TSMixerSeqCls
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from typing import Any, Dict, List, Optional
import argparse
import warnings
import numpy as np
import random
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
class TSMixerSeqClsTrainModule(pl.LightningModule):
    def __init__(self, optimizer_cfg: DictConfig, model_cfg: DictConfig, model_name: str, **kwargs):
        super(TSMixerSeqClsTrainModule, self).__init__(**kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = TSMixerSeqCls(
            model_cfg.bottleneck,
            model_cfg.mixer,
            model_cfg.sequence_cls,
            model_name
        )

    def metrics(self, outputs):
        ACC, Macro_Precision, Macro_F1, Macro_recall, length = 0, 0, 0, 0, len(outputs)
        for output in outputs:
            ACC += accuracy_score(output['targets'], output['predicts'])
            Macro_Precision += precision_score(output['targets'], output['predicts'], average='macro')
            Macro_F1 += f1_score(output['targets'], output['predicts'], average='macro')
            Macro_recall += recall_score(output['targets'], output['predicts'], average='macro')
        return {
            "ACC": ACC / length,
            "Macro_Precision": Macro_Precision / length,
            "Macro_F1": Macro_F1 / length,
            "Macro_Recall": Macro_recall / length
        }

    def shared_step(self, batch, batch_idx):
        x, targets = batch['inputs'], batch["targets"].to(torch.int64)
        outs = self.model(x)
        loss = F.cross_entropy(outs, targets)
        predict = torch.argmax(outs, dim=1)
        return {
            'batch_ids': batch_idx,
            'predicts': predict,#.detach().cpu().numpy(),
            'targets': targets,#.detach().cpu().numpy(),
            'loss': loss
        }

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch, batch_idx)
        self.log('train_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return results

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accuracy = self.metrics(outputs)
        self.log('train_acc', accuracy['ACC'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        results = self.shared_step(batch, batch_idx)
        # self.log('val_loss', results['loss'], on_step=False, on_epoch=False, prog_bar=False, logger=True)
        return results

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accuracy = self.metrics(outputs)
        self.log('val_acc', accuracy['ACC'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pre', accuracy["Macro_Precision"], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_f1', accuracy["Macro_F1"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', accuracy['Macro_Recall'], on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        results = self.shared_step(batch, batch_idx)
        self.log('test_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return results

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accuracy = self.metrics(outputs)
        self.log('test_acc', accuracy['ACC'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_pre', accuracy["Macro_Precision"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', accuracy["Macro_F1"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', accuracy['Macro_Recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_cfg)
        return optimizer

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--cfg', type=str)
    args.add_argument('-n', '--name', type=str)
    args.add_argument('-a', '--architecture', type=str)
    args.add_argument('-p', '--ckpt', type=str)
    args.add_argument('-m', '--mode', type=str, default='train')
    return args.parse_args()

def get_module_cls(type: str):
    if type in ['imdb','20NG', 'MSRP','BBC','xnli','yelp5','mrpc','amazon5','rte','snli','ag_news','sst2','qnli','cola','yelp','qqp','hyperpartisan','dbpedia','amazon']:
        return TSMixerSeqClsTrainModule
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
if __name__ == '__main__':
    seed = 1024
    set_seed(seed)
    args = parse_args()
    cfg = OmegaConf.load('cfg/{}.yml'.format(args.cfg))
    vocab_cfg = cfg.vocab
    train_cfg = cfg.train
    model_cfg = cfg.model
    module_cls = get_module_cls(train_cfg.type)
    if args.ckpt:
        train_module = module_cls.load_from_checkpoint(args.ckpt, model_name=args.architecture,
                                                       optimizer_cfg=train_cfg.optimizer, model_cfg=model_cfg)
    else:
        train_module = module_cls(model_name=args.architecture, optimizer_cfg=train_cfg.optimizer, model_cfg=model_cfg)
    data_module = TSMixerDataModule(cfg.vocab, train_cfg, model_cfg.projection)
    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_acc',
                # filename=args.architecture+'-best-{epoch:03d}-{val_acc:.4f}-{val_f1:.4f}',
                filename=args.architecture+'-best-{epoch:03d}-{val_acc:.4f}-{val_f1:.4f}',
                save_top_k=1,
                mode='max',
                save_last=True
            )
        ],
        enable_checkpointing=True,
        # gpus=-1,
        #precision=16,
        #plugins='ddp_sharded',
        log_every_n_steps=train_cfg.log_interval_steps,
        logger=pl.loggers.TensorBoardLogger(train_cfg.tensorboard_path, args.name),
        max_epochs=train_cfg.epochs,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1
    )
    if args.mode == 'train':
        trainer.fit(train_module, data_module)
    if args.mode == 'test':
        trainer.test(train_module, data_module)
