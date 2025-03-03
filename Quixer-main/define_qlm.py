import random
import os
import time
import math
from tqdm import tqdm
from typing import Any, Optional, Tuple, Callable

import torchtext; torchtext.disable_torchtext_deprecation_warning()
import numpy as np

import torch
import torchtext

from model import Quixer
from ctransformer import Transformer, LSTM, FNet


from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer


def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def batchify_s2s(data: torch.Tensor, batch_size, bptt, pad_token, device) -> torch.Tensor:
    seq_len = (data.size(0) - 1) // batch_size
    batched_data = data[: seq_len * batch_size].view(batch_size, seq_len).T

    # Take last BPTT elements for all but the last batch
    bptt_data = torch.cat((torch.full((bptt, 1), pad_token, device=device), batched_data[-bptt:, :-1]), dim=1)

    return torch.cat((bptt_data, batched_data))


def init_weights(model: torch.nn.Module) -> None:
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    model.apply(_init_weights)
    
def read_file(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    # 如果是训练数据文件，则打印前三行
    if file_path == 'ptb_train.txt':
        print("训练数据文件的前三行：")
        for i, line in enumerate(lines[:3]):
            print(f"第 {i+1} 行：{line}")
    return lines

# 在 setup_dataset 中调用
def setup_dataset(device: torch.device, batch_size: int, bptt: int) -> Tuple:
    # 从本地文件读取数据
    train_sents = read_file('ptb_train.txt')
    val_sents = read_file('ptb_valid.txt')
    test_sents = read_file('ptb_test.txt')
    
    print(f"测试数据总行数: {len(test_sents)}")
    if len(test_sents) > 0:
        print("测试数据的前三行：")
        for i, line in enumerate(test_sents[:3]):
            print(f"第 {i+1} 行：{line}")

    tokenizer = get_tokenizer("basic_english")
    
    # 根据训练数据构建词汇表
    vocab = build_vocab_from_iterator(map(tokenizer, train_sents), specials=["<pad>", "<unk>", "<eos>"])
    vocab.set_default_index(vocab["<unk>"])
    PAD_TOK = vocab["<pad>"]

    def data_process(raw_text_iter) -> torch.Tensor:
        data = [
            torch.tensor(vocab(tokenizer(item)) + [vocab["<eos>"]], dtype=torch.long)
            for item in raw_text_iter
        ]
        filtered_data = tuple(filter(lambda t: t.numel() > 1, data))
        if not filtered_data:
            raise ValueError("数据处理后没有有效数据，请检查输入文件。")
        return torch.cat(filtered_data).to(device)
    
    # 将各个数据集扁平化成一个长序列
    train_flat = data_process(train_sents)
    val_flat = data_process(val_sents)
    test_flat = data_process(test_sents)
    
    # 使用预定义的 batchify_s2s 函数来构造 (x, y) 对
    train_iter = batchify_s2s(train_flat, batch_size * bptt, bptt, PAD_TOK, device)
    val_iter = batchify_s2s(val_flat, batch_size * bptt, bptt, PAD_TOK, device)
    test_iter = batchify_s2s(test_flat, batch_size * bptt, bptt, PAD_TOK, device)
    
    return vocab, (train_iter, val_iter, test_iter), PAD_TOK





def get_batch_s2s(source, i, BPTT, *args):
    return source[i: i + BPTT].T, source[i + BPTT]


def create_model(
    hyperparams: dict[str, Any], device: torch.device, vocab_size: int
) -> torch.nn.Module:
    model_str = hyperparams["model"]
    if model_str == "QLINSVT":
        model = Quixer(
            n_qubits=hyperparams["qubits"],
            n_words=hyperparams["window"],
            degree=hyperparams["layers"],
            n_ansatz_layers=hyperparams["ansatz_layers"],
            vocab_size=vocab_size,
            embedding_dim=hyperparams["dimension"],
            dropout=hyperparams["dropout"],
            device=device,
        )
    elif model_str == "FNet":
        model = FNet(
            vocab_size=vocab_size,
            emb_dim=hyperparams["dimension"],
            hid_dim=4 * hyperparams["dimension"],
            n_layers=hyperparams["layers"],
            dropout=hyperparams["dropout"],
        )
    elif model_str == "VAS":
        model = Transformer(
            emb_dim=hyperparams["dimension"],
            hid_dim=4 * hyperparams["dimension"],
            n_heads=hyperparams["heads"],
            n_layers=hyperparams["layers"],
            vocab_size=vocab_size,
            dropout=hyperparams["dropout"]
        )
    elif model_str == "LSTM":
        model = LSTM(
            emb_dim=hyperparams["dimension"],
            hid_dim=hyperparams["dimension"],
            n_layers=hyperparams["layers"],
            vocab_size=vocab_size,
            dropout=hyperparams["dropout"]
        )
    else:
        raise ValueError(f"Unrecognized model: {model_str}")

    return model


def train_epoch(
    model: torch.nn.Module,
    iterator: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion,
    clip: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    print_iter: int,
    window_size: int,
    pad_token: int,
    device: torch.device,
    batch_size: int,
):
    model.train()

    epoch_loss = 0

    n_batches = iterator.shape[0] - window_size

    idxs = list(range(n_batches))
    random.shuffle(idxs)

    for ctr, batch_idx in tqdm(enumerate(idxs), total=n_batches):
        x, y = get_batch_s2s(
            iterator, batch_idx, window_size, device, batch_size
        )
        optimizer.zero_grad()

        yhat, norm_avg = model(x)

        loss = criterion(yhat, y)
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / n_batches


def evaluate(
    model: torch.nn.Module,
    iterator,
    criterion,
    window_size: int,
    pad_token: int,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()

    epoch_loss = 0

    n_batches = iterator.shape[0] - window_size

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches)):
            x, y = get_batch_s2s(
                iterator, batch_idx, window_size, device, batch_size
            )

            yhat, _ = model(x)

            loss = criterion(yhat, y)

            epoch_loss += loss.item()

    return epoch_loss / n_batches

def train_cycle(
    model: torch.nn.Module,
    hyperparams: dict[str, Any],
    device: torch.device,
    train_iter: torch.Tensor,
    val_iter: torch.Tensor,
    test_iter: torch.Tensor,
    pad_token: int,
) -> float:

    checkpoint_fpath = f"./trained_models/q_transformer_lm_{hyperparams['model']}_{hyperparams['seed']}_{int(time.time())}.pt"

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
    )

    scheduler = None
    if hyperparams["lr_sched"] == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=hyperparams["restart_epochs"]
        )

    criterion = torch.nn.CrossEntropyLoss()

    def _evaluate(iter: torch.Tensor):
        return evaluate(
            model,
            iter,
            criterion,
            hyperparams["window"],
            pad_token,
            device,
            hyperparams["batch_size"],
        )

    best_valid_loss = float("inf")
    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_iter,
            optimizer,
            criterion,
            hyperparams["max_grad_norm"],
            scheduler,
            hyperparams["print_iter"],
            hyperparams["window"],
            pad_token,
            device,
            hyperparams["batch_size"],
        )

        valid_loss = _evaluate(val_iter)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint_fpath = f"./trained_models/q_transformer_lm_{hyperparams['model']}_{hyperparams['seed']}_{int(time.time())}.pt"
    # 确保保存模型的目录存在
            os.makedirs(os.path.dirname(checkpoint_fpath), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_fpath)
            

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train ppl: {math.exp(train_loss)}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss)}")

    model.load_state_dict(torch.load(checkpoint_fpath))

    valid_loss = _evaluate(val_iter)
    test_loss = _evaluate(test_iter)

    print("FINAL TRAINED MODEL STATS:")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss)}")
    print(f"\t Test Loss: {test_loss:.3f} |  Test ppl: {math.exp(test_loss)}")

    return test_loss


def seed(SEED: int) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def get_train_evaluate(device: torch.device) -> Callable:
    def train_evaluate(parameterization: dict[str, Any]) -> float:
        """
        Train the model and then compute an evaluation metric.
        """

        if "seed" not in parameterization:
            parameterization["seed"] = int.from_bytes(os.urandom(4), "big")

        seed(parameterization["seed"])

        vocab, (train_iter, val_iter, test_iter), PAD_TOK = setup_dataset(
            device,
            parameterization["batch_size"],
            parameterization["window"]
        )

        model = create_model(parameterization, device, len(vocab))

        init_weights(model)

        model = model.to(device)

        valid_loss = train_cycle(
            model,
            parameterization,
            device,
            train_iter,
            val_iter,
            test_iter,
            PAD_TOK,
        )

        return valid_loss

    return train_evaluate

