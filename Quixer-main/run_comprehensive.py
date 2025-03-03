
import json
import math
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Any

from define_qlm import get_train_evaluate
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Tuple
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# 设备选择
device_name = sys.argv[1] if len(sys.argv) >= 3 else 'cpu'
device = torch.device(device_name)
print(f"Running on device: {device}")



# 定义自定义数据集类（支持传入词汇表）
class TextDataset(Dataset):
    def __init__(self, file_path: str, vocab=None, tokenizer=None):
        self.file_path = file_path
        self.tokenizer = tokenizer or (lambda x: x.split())
        self.data = self.load_data()
        if vocab is None:
            # 这里可以自动构造一个简单词汇表，或抛出异常提示用户传入
            from collections import Counter
            word_freq = Counter(" ".join(self.data).split())
            from torchtext.vocab import build_vocab_from_iterator
            vocab = build_vocab_from_iterator([word_freq.keys()], specials=['<unk>', '<pad>', '<eos>'])
            vocab.set_default_index(vocab['<unk>'])
        self.vocab = vocab


    def load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # 每一行作为一个样本，并去掉空行
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokenized_text = self.tokenizer(text)
        # 将单词转换为对应的词汇索引，并在末尾添加 <eos> 标记
        indices = self.vocab(tokenized_text) + [self.vocab["<eos>"]]
        return torch.tensor(indices, dtype=torch.long)
    
from collections import Counter
from torchtext.vocab import Vocab


from collections import Counter
from torchtext.vocab import build_vocab_from_iterator
# 创建词汇表
with open('ptb_train.txt', 'r', encoding='utf-8') as f:
    text = f.read().split()

word_freq = Counter(text)

# 定义一个生成器，返回训练集所有单词（分词后的结果）
def yield_tokens():
    for token in word_freq.keys():
        yield token

# 构造词汇表，指定特殊标记
vocab = build_vocab_from_iterator([list(yield_tokens())], specials=['<unk>', '<pad>', '<sos>', '<eos>'])
vocab.set_default_index(vocab['<unk>'])

# 重新创建数据集
train_dataset = TextDataset('ptb_train.txt', vocab)

# 构建词汇表和数据集的函数
def setup_dataset(device: torch.device, batch_size: int, bptt: int) -> Tuple:
    # 从本地文件读取数据
    def read_file(file_path: str) -> list:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    train_sents = read_file('ptb_train.txt')
    val_sents = read_file('ptb_valid.txt')
    test_sents = read_file('ptb_test.txt')

    # 定义分词器
    tokenizer = get_tokenizer("basic_english")

    # 根据训练数据构建词汇表
    vocab = build_vocab_from_iterator(map(tokenizer, train_sents),
                                      specials=["<pad>", "<unk>", "<eos>"])
    vocab.set_default_index(vocab["<unk>"])
    PAD_TOK = vocab["<pad>"]

    # 创建数据集，传入词汇表
    train_dataset = TextDataset('ptb_train.txt', vocab, tokenizer)
    val_dataset = TextDataset('ptb_valid.txt', vocab, tokenizer)
    test_dataset = TextDataset('ptb_test.txt', vocab, tokenizer)

    # 定义数据加载器，使用 pad_sequence 来补齐不同长度的序列
    collate_fn = lambda batch: torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=PAD_TOK)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
"""
# 定义自定义数据集类
class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer=None):
        self.file_path = file_path
        self.tokenizer = tokenizer or (lambda x: x.split())  # 默认按空格分词
        self.data = self.load_data()

    def load_data(self):
        with open(self.file_path, 'r') as f:
            # 假设文件中每一行是一个样本
            return [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # 分词和转换为Tensor
        tokenized_text = self.tokenizer(text)
        # 如果需要将token转为数字ID，可以在此处使用词汇表进行转换
        return torch.tensor([len(word) for word in tokenized_text], dtype=torch.long)
"""
# 加载数据集
train_dataset = TextDataset('ptb_train.txt')
valid_dataset = TextDataset('ptb_valid.txt')
test_dataset = TextDataset('ptb_test.txt')

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True))
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True))

# 超参数
quixer_hparams = {
    "qubits": 6,
    "layers": 3,
    "ansatz_layers": 4,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "QLINSVT",
    "print_iter": 50
}

lstm_hparams = {
    "layers": 2,
    "window": 32,
    "residuals": False,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.30,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "print_iter": 50
}

fnet_hparams = {
    "layers": 2,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "FNet",
    "print_iter": 50
}

vas_hparams = {
    "layers": 1,
    "heads": 1,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.001,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "VAS",
    "print_iter": 50
}

cdimensions = [96, 128]
qdimensions = [512]

model_map = {
    "VAS": (vas_hparams, cdimensions),
    "LSTM": (lstm_hparams, cdimensions),
    "FNet": (fnet_hparams, cdimensions),
    "QLINSVT": (quixer_hparams, qdimensions)
}

torch.backends.cudnn.deterministic = True

train_evaluate = get_train_evaluate(device)



# 训练并评估模型
for model_name, meta in model_map.items():
    fix_hyperparams, dimensions = meta
    for dim in dimensions:
        for seed in torch.randint(high=1000000, size=(10,)).tolist():
            fix_hyperparams["model"] = model_name
            fix_hyperparams["dimension"] = dim
            fix_hyperparams["seed"] = seed

            # 将数据加载器传递给训练函数
            fix_hyperparams["train_loader"] = train_loader
            fix_hyperparams["valid_loader"] = valid_loader
            fix_hyperparams["test_loader"] = test_loader     
            train_evaluate(fix_hyperparams)
