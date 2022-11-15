
from TSMixer import TS_Mixer
from classification import SequenceClassificationLayer
from mixer import Mixer
from omegaconf.dictconfig import DictConfig
import torch
import torch.nn as nn

class TSMixerSeqCls(nn.Module):
    def __init__(
        self,
        bottleneck_cfg: DictConfig,
        mixer_cfg: DictConfig,
        seq_cls_cfg: DictConfig,
        model_name: str,
        **kwargs
    ):
        super(TSMixerSeqCls, self).__init__(**kwargs)
        self.mixer_cfg = mixer_cfg
        self.ts_mixer = TSMixer(bottleneck_cfg, mixer_cfg, model_name)
        self.seq_cls = SequenceClassificationLayer(**seq_cls_cfg)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) == 2:
            u, v = torch.chunk(inputs, 2, dim=1)
            u = self.ts_mixer(u.squeeze(1))
            v = self.ts_mixer(v.squeeze(1))
            reprs = torch.cat((u, v, torch.abs(u-v)), dim=1)
        else:
            reprs = self.ts_mixer(inputs)
        seq_logits = self.seq_cls(reprs)
        return seq_logits

class TSMixer(nn.Module):
    def __init__(
        self,
        bottleneck_cfg: DictConfig,
        mixer_cfg: DictConfig,
        model_name: str,
        **kwargs
    ):
        super(TSMixer, self).__init__(**kwargs)
        self.bottleneck = nn.Linear(bottleneck_cfg.feature_size, bottleneck_cfg.hidden_dim)
        if model_name == "mixer":
            self.mixer = Mixer(**mixer_cfg)
        elif model_name == "TS-Mixer":
            self.mixer = TS_Mixer(**mixer_cfg)
        else:
            raise ValueError("Unkown model: {}".format(model_name))
        self.pp=self.bottleneck
        self.pipeline = nn.Sequential(
            self.bottleneck,
            self.mixer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) == 2:
            sentence_first, sentence_second = torch.chunk(inputs, 2, dim=1)
            sentence_first = sentence_first.squeezce(1)
            sentence_second = sentence_second.squeeze(1)
            u = self.pipeline(sentence_first)
            v = self.pipeline(sentence_second)
            return torch.cat(u, v, torch.abs(u-v), dim=1)
        else:
            reprs = self.pipeline(inputs)
        return reprs
