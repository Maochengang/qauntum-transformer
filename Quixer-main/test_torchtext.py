import torch
from torch.utils.data import Dataset

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Tuple
from define_qlm import batchify_s2s

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Tuple
import sys
import os
sys.path.append(os.path.abspath(".."))
import torchtext; 
torchtext.disable_torchtext_deprecation_warning()

# 定义自定义数据集类（支持传入词汇表）
class TextDataset(Dataset):
    def __init__(self, file_path: str, vocab, tokenizer=None):
        self.file_path = file_path
        self.tokenizer = tokenizer or (lambda x: x.split())  # 默认按空格分词
        self.vocab = vocab  # 词汇表对象
        self.data = self.load_data()

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

    # 这里假设你已有一个 batchify_s2s 函数处理批次进一步处理（例如拼接为长序列），
    # 如果不需要可直接使用 DataLoader 输出的 batch。

    # 示例：将 DataLoader 输出转换为一个长张量（根据具体需求调整）
    def flatten_loader(loader):
        all_data = []
        for batch in loader:
            all_data.append(batch.view(-1))
        return torch.cat(all_data)

    train_flat = flatten_loader(train_loader)
    val_flat = flatten_loader(val_loader)
    test_flat = flatten_loader(test_loader)

    # 使用预定义的 batchify_s2s 函数构造 (x, y) 对（假设该函数已正确定义）
    train_iter = batchify_s2s(train_flat, batch_size * bptt, bptt, PAD_TOK, device)
    val_iter = batchify_s2s(val_flat, batch_size * bptt, bptt, PAD_TOK, device)
    test_iter = batchify_s2s(test_flat, batch_size * bptt, bptt, PAD_TOK, device)

    return vocab, (train_iter, val_iter, test_iter), PAD_TOK

# 设备选择
device_name = sys.argv[1] if len(sys.argv) >= 2 else 'cpu'
device = torch.device(device_name)
print(f"Running on device: {device}")

# 下面你可以调用 setup_dataset() 获取数据，然后传递给训练函数

"""
class PTBDataset(Dataset):
    def __init__(self, train_file, valid_file, test_file):
        # 打开并读取文件
        with open(train_file, 'r') as f:
            self.train_data = f.readlines()

        with open(valid_file, 'r') as f:
            self.valid_data = f.readlines()

        with open(test_file, 'r') as f:
            self.test_data = f.readlines()

    def __len__(self):
        return len(self.train_data)  # 假设训练集数据是最大长度，或可以根据需求修改

    def __getitem__(self, idx):
        # 这里可以根据需要返回不同的数据，训练、验证或测试数据
        return {
            "train": self.train_data[idx],
            "valid": self.valid_data[idx],
            "test": self.test_data[idx]
        }

# 假设文件路径如下
train_file = 'ptb_train.txt'
valid_file = 'ptb_valid.txt'
test_file = 'ptb_test.txt'

# 创建数据集实例
dataset = PTBDataset(train_file, valid_file, test_file)

# 访问训练集的一个样本
print(dataset[0]["train"])  # 查看训练数据中的第一行
"""