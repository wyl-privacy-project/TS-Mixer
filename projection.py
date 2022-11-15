from typing import List
import argparse
import numpy as np
import hashlib
from omegaconf import OmegaConf

SENTENCEPIECE_IS_CONTINUATION = lambda t: not t.startswith("▁")
WORDPIECE_IS_CONTINUATION = lambda t: t.startswith("##")
MAX_HASH_VALUE = 2 ** 31 - 1
cfg_path= 'cfg/MY_MLP.yml'

class Projection:
    def __init__(
            self, hash_path: str, feature_size: int, **kwargs
    ):
        self.hash = CachedHash(hash_path)
        self.cbf = CountingBloomFilter(feature_size)
        self.feature_size = feature_size

    def __call__(self, words: List[List[str]]) -> np.ndarray:
        hashed = np.array([np.array([self.hash(token) for token in word]).min(axis=-2) for word in words])
        features = self.cbf(hashed)
        return features


class MinHash:
    def __init__(self, num_hashes: int, ngram_size: int):
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size
        self.hash_fn1 = lambda data: int.from_bytes(hashlib.new('sha256', data.encode("utf8")).digest(), 'little')
        self.hash_fn2 = lambda data: int.from_bytes(hashlib.new('sha224', data.encode("utf8")).digest(), 'little')

    def __call__(self, token: str, is_cont: bool) -> np.ndarray:
        if is_cont or len(token) < self.ngram_size + 1:
            hash1 = self.hash_fn1(token)
            hash2 = self.hash_fn2(token)
            hash = np.array([(hash1 + i * hash2) % MAX_HASH_VALUE for i in range(self.num_hashes)])
            return hash
        ngrams = []
        for index in range(len(token) - self.ngram_size + 1):
            hash1 = self.hash_fn1(token[index:index + self.ngram_size])
            hash2 = self.hash_fn2(token[index:index + self.ngram_size])
            hash = np.array([(hash1 + i * hash2) % MAX_HASH_VALUE for i in range(self.num_hashes)])
            ngrams.append(hash)
        fingerprint = np.array(ngrams).min(axis=-2)
        return fingerprint


class CachedHash:
    def __init__(self, path: str):
        self.cached_hash = np.load(path, allow_pickle=True).item()

    def __call__(self, token: str) -> np.ndarray:
        return self.cached_hash[token]


class CountingBloomFilter:
    def __init__(self, feature_size: int):
        self.feature_size = feature_size
        self.one_hot = np.eye(feature_size, dtype=np.float32)

    def __call__(self, words: np.ndarray) -> np.ndarray:
        features = self.one_hot[words % self.feature_size].sum(axis=-2)
        return features

