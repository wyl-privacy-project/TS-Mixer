from tokenizers.implementations import BertWordPieceTokenizer, SentencePieceBPETokenizer, SentencePieceUnigramTokenizer
import re
from projection import Projection
import numpy as np
from datasets import load_dataset
import pytorch_lightning as pl
from omegaconf.dictconfig import DictConfig
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from typing import Any, Dict, List
import torch

class TSMixerDataModule(pl.LightningDataModule):

    def __init__(self, vocab_cfg: DictConfig, train_cfg: DictConfig, proj_cfg: DictConfig, **kwargs):
        super(TSMixerDataModule, self).__init__(**kwargs)
        self.vocab_cfg = vocab_cfg
        self.train_cfg = train_cfg
        self.projecion = Projection(vocab_cfg.vocab_path, proj_cfg.feature_size)

        if vocab_cfg.tokenizer_type == 'wordpiece':
            self.tokenizer = BertWordPieceTokenizer(**vocab_cfg.tokenizer)
        if vocab_cfg.tokenizer_type == 'sentencepiece_bpe':
            self.tokenizer = SentencePieceBPETokenizer(**vocab_cfg.tokenizer)
        if vocab_cfg.tokenizer_type == 'sentencepiece_unigram':
            self.tokenizer = SentencePieceUnigramTokenizer(**vocab_cfg.tokenizer)

    def get_dataset_cls(self):
        the_data = globals()[self.train_cfg.dataset_type]
        return the_data

    def setup(self, stage: str = None):
        label_list = Path(self.train_cfg.labels).read_text().splitlines() if isinstance(self.train_cfg.labels,                                                                              str) else self.train_cfg.labels
        self.label_map = {label: index for index, label in enumerate(label_list)}
        dataset_cls = self.get_dataset_cls()
        if stage in (None, 'fit'):
            self.train_set = dataset_cls( 'train', self.train_cfg.max_seq_len, self.tokenizer, self.projecion,
                                         self.label_map)
            if self.train_cfg.dataset_type in ['AGDataset', 'ImdbDataset', "YelpDataset",  'DbpediaDataset',
                                               'AmazonDataset']:
                mode = 'test'
            else:
                mode = 'validation'
            self.eval_set = dataset_cls(mode, self.train_cfg.max_seq_len, self.tokenizer, self.projecion,
                                        self.label_map)
        if stage in (None, 'test'):
            self.test_set = dataset_cls('test', self.train_cfg.max_seq_len, self.tokenizer, self.projecion,
                                        self.label_map)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.train_cfg.train_batch_size, shuffle=True,
                          num_workers=self.train_cfg.num_workers, persistent_workers=True)  # , pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_set, self.train_cfg.test_batch_size, shuffle=True,
                          num_workers=self.train_cfg.num_workers, persistent_workers=True)  # , pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.train_cfg.test_batch_size, shuffle=True,
                          num_workers=self.train_cfg.num_workers, persistent_workers=True)  # , pin_memory=True)

class TSMixerDataset(Dataset):
    def __init__(self, max_seq_len: int, tokenizer: Tokenizer, projection: Projection, label_map: Dict[str, int],
                 **kwargs):
        super(TSMixerDataset, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.projection = projection
        self.max_seq_len = max_seq_len
        self.label_map = label_map

    def normalize(self, text: str) -> str:
        return text.replace('’', '\'') \
            .replace('–', '-') \
            .replace('‘', '\'') \
            .replace('´', '\'') \
            .replace('“', '"') \
            .replace('”', '"')

    def project_features(self, words: List[str]) -> np.ndarray:
        encoded = self.tokenizer.encode(words, is_pretokenized=True, add_special_tokens=False)
        tokens = [[] for _ in range(len(words))]
        for index, token in zip(encoded.words, encoded.tokens):
            tokens[index].append(token)
        features = self.projection(tokens)
        padded_featrues = np.pad(features, ((0, self.max_seq_len - len(words)), (0, 0)))
        return padded_featrues

    def get_words(self, fields: Dict) -> List[str]:
        raise NotImplementedError

    def compute_labels(self, fields: Dict) -> np.ndarray:
        raise NotImplementedError

    def __getitem__(self, index) -> Dict[str, Any]:
        fields = self.data[index]
        words = self.get_words(fields)
        features = self.project_features(words)
        labels = self.compute_labels(fields)
        return {
            'inputs': features,
            'targets': labels
        }

class AmazonDataset(TSMixerDataset):
    def __init__(self, filename: str, *args, **kwargs) -> None:
        super(AmazonDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset("amazon_polarity", split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !") \
            .replace("\n", " ")

    def get_words(self, fields: Dict) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["content"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: Dict) -> np.ndarray:
        return np.array(fields["label"])

class HyperpartisanDataset(TSMixerDataset):
    def __init__(self, filename: str, *args, **kwargs) -> None:
        super(HyperpartisanDataset, self).__init__(*args, **kwargs)
        data = load_dataset("hyperpartisan_news_detection", "byarticle", split="train")
        # data = load_dataset('hyperpartisan_news_detection', 'byarticle')
        # data = data['train'].train_test_split(test_size=0.2,shuffle=False)
        # self.data = data[filename]
        # print(data)
        train_size = int(len(data) * 0.8)
        val_size = int(len(data) * 0.1)
        test_size = len(data)-train_size-val_size
        train, val, test = torch.utils.data.random_split(data, [train_size, val_size, test_size])
        if filename == "train":
            self.data = train
        if filename == "test":
            self.data = test
        if filename == 'validation':
            self.data = val
        self.label_map = {False: 0, True: 1}

    def __len__(self) -> int:
        return len(self.data)

    def len_compute(self):
        return

    def normalize(self, text: str) -> str:
        html_label = "<[^>]+>"
        email = "^[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)"
        chars = "&#[0-9]*"
        url = "(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?"
        text = text.replace('’', '\'') \
            .replace('–', '-') \
            .replace('‘', '\'') \
            .replace('´', '\'') \
            .replace('“', '"') \
            .replace('”', '"') \
            .replace('<splt>', '  ').replace('&#[0-9]*; ', "")
        text = re.sub(html_label, "", text)
        text = re.sub(url, "", text)
        text = re.sub(email, "", text)
        text = re.sub(chars, "", text)
        return text

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["text"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(self.label_map[fields["hyperpartisan"]])

class QQPDataset(TSMixerDataset):
    def __init__(self,filename: str, *args, **kwargs) -> None:
        super(QQPDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('glue', 'qqp', split=filename)

    def __len__(self):
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !")

    def get_words(self, fields: Dict) -> List:
        return [[w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["question1"]))][
                :self.max_seq_len],
                [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["question2"]))][
                :self.max_seq_len]]

    def compute_labels(self, fields):
        return np.array(fields["label"])

    def __getitem__(self, index) -> Dict[str, Any]:
        fields = self.data[index]
        words = self.get_words(fields)
        u = self.project_features(words[0]).reshape(1, self.max_seq_len, -1)
        v = self.project_features(words[1]).reshape(1, self.max_seq_len, -1)
        features = np.concatenate((u, v), axis=0)
        labels = self.compute_labels(fields)
        return {
            'inputs': features,
            'targets': labels
        }

class YelpDataset(TSMixerDataset):
    def __init__(self,  filename: str, *args, **kwargs) -> None:
        super(YelpDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('yelp_polarity', 'plain_text', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !") \
            .replace("\n", " ")

    def get_words(self, fields: Dict) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["text"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: Dict) -> np.ndarray:
        return np.array(fields["label"])

class DbpediaDataset(TSMixerDataset):
    def __init__(self, filename: str, *args, **kwargs) -> None:
        super(DbpediaDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('dbpedia_14', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ').replace('’', '\'') \
            .replace('–', '-') \
            .replace('‘', '\'') \
            .replace('´', '\'') \
            .replace('“', '"') \
            .replace('”', '"')

    def get_words(self, fields: Dict) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["content"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: Dict) -> np.ndarray:
        return np.array(fields["label"])

class ImdbDataset(TSMixerDataset):
    def __init__(self,  filename: str, *args, **kwargs) -> None:
        super(ImdbDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('imdb', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: Dict) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["text"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: Dict) -> np.ndarray:
        return np.array(fields["label"])

class SST2Dataset(TSMixerDataset):
    def __init__(self, filename: str, *args, **kwargs) -> None:
        super(SST2Dataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('glue', 'sst2', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ').replace('’', '\'') \
            .replace('–', '-') \
            .replace('‘', '\'') \
            .replace('´', '\'') \
            .replace('“', '"') \
            .replace('”', '"')

    def get_words(self, fields: Dict) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["sentence"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: Dict) -> np.ndarray:
        return np.array(fields["label"])

class ColaDataset(TSMixerDataset):
    def __init__(self,  filename: str, *args, **kwargs) -> None:
        super(ColaDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('glue', 'cola', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: Dict) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["sentence"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: Dict) -> np.ndarray:
        return np.array(fields["label"])

class AGDataset(TSMixerDataset):
    def __init__(self, filename: str, *args, **kwargs) -> None:
        super(AGDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('ag_news', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ') \
            .replace('’', '\'') \
            .replace('–', '-') \
            .replace('‘', '\'') \
            .replace('´', '\'') \
            .replace('“', '"') \
            .replace('”', '"')

    def get_words(self, fields: Dict) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["text"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: Dict) -> np.ndarray:
        return np.array(fields["label"])

class QNLIDataset(TSMixerDataset):
    def __init__(self, filename: str, *args, **kwargs) -> None:
        super(QNLIDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('glue', 'qnli', split=filename)

    def __len__(self):
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !")

    def get_words(self, fields: Dict) -> List:
        return [[w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["question"]))][
                :self.max_seq_len],
                [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["sentence"]))][
                :self.max_seq_len]]

    def compute_labels(self, fields):
        return np.array(fields["label"])

    def __getitem__(self, index) -> Dict[str, Any]:
        fields = self.data[index]
        words = self.get_words(fields)
        u = self.project_features(words[0]).reshape(1, self.max_seq_len, -1)
        v = self.project_features(words[1]).reshape(1, self.max_seq_len, -1)
        features = np.concatenate((u, v), axis=0)
        labels = self.compute_labels(fields)
        return {
            'inputs': features,
            'targets': labels
        }

class SNLIDataset(TSMixerDataset):
    def __init__(self, filename: str, *args, **kwargs) -> None:
        super(SNLIDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('snli', split=filename)

    def __len__(self):
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !")

    def get_words(self, fields: Dict) -> List:
        return [[w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["premise"]))][
                :self.max_seq_len],
                [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["hypothesis"]))][
                :self.max_seq_len]]

    def compute_labels(self, fields):
        return np.array(fields["label"] + 1)

    def __getitem__(self, index) -> Dict[str, Any]:
        fields = self.data[index]
        words = self.get_words(fields)
        u = self.project_features(words[0]).reshape(1, self.max_seq_len, -1)
        v = self.project_features(words[1]).reshape(1, self.max_seq_len, -1)
        features = np.concatenate((u, v), axis=0)
        labels = self.compute_labels(fields)
        return {
            'inputs': features,
            'targets': labels
        }

class RTEDataset(TSMixerDataset):
    def __init__(self, filename: str, *args, **kwargs) -> None:
        super(RTEDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('glue', 'rte', split=filename)

    def __len__(self):
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !")

    def get_words(self, fields: Dict) -> List:
        return [[w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields['sentence1']))][
                :self.max_seq_len],
                [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields['sentence2']))][
                :self.max_seq_len]]

    def compute_labels(self, fields):
        return np.array(fields["label"])

    def __getitem__(self, index) -> Dict[str, Any]:
        fields = self.data[index]
        words = self.get_words(fields)
        u = self.project_features(words[0]).reshape(1, self.max_seq_len, -1)
        v = self.project_features(words[1]).reshape(1, self.max_seq_len, -1)
        features = np.concatenate((u, v), axis=0)
        labels = self.compute_labels(fields)
        return {
            'inputs': features,
            'targets': labels
        }
