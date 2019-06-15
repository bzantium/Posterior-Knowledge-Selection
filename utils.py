import torch
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader, Dataset


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def build_vocab(path, vocab_size):
    with open(path, errors="ignore") as file:
        X = []
        K = []
        y = []
        k = []
        word_counter = Counter()
        vocab = dict()
        reverse_vocab = dict()
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        vocab['<SOS>'] = 2
        vocab['<EOS>'] = 3
        vocab_idx = len(vocab)

        for line in file:
            dialog_id = line.split()[0]
            if dialog_id == "1":
                k = []

            if "your persona:" in line:
                if len(k) == 3:
                    continue
                k_line = line.split("persona:")[1].strip("\n")
                k.append(k_line)
                for word in k_line.split():
                    if word in vocab:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1

            elif "__SILENCE__" not in line:
                K.append(k)
                X_line = " ".join(line.split("\t")[0].split()[1:])
                y_line = line.split("\t")[1].strip("\n")
                X.append(X_line)
                y.append(y_line)

                for word in X_line.split():
                    if word in vocab:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1

                for word in y_line.split():
                    if word in vocab:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1

        for key, _ in word_counter.most_common(vocab_size - 2):
            vocab[key] = vocab_idx
            vocab_idx += 1

        for key, value in vocab.items():
            reverse_vocab[value] = key

    return vocab, reverse_vocab


def load_dataset(path, vocab):
    with open(path, errors="ignore") as file:
        X = []
        K = []
        y = []
        k = []

        for line in file:
            dialog_id = line.split()[0]
            if dialog_id == "1":
                k = []

            if "your persona:" in line:
                if len(k) == 3:
                    continue
                k_line = line.split("persona:")[1].strip("\n")
                k.append(k_line)

            elif "__SILENCE__" not in line:
                K.append(k)
                X_line = " ".join(line.split("\t")[0].split()[1:])
                y_line = line.split("\t")[1].strip("\n")
                X.append(X_line)
                y.append(y_line)

    X_ind = []
    y_ind = []
    K_ind = []

    for line in X:
        X_temp = []
        for word in line.split():
            if word in vocab:
                X_temp.append(vocab[word])
            else:
                X_temp.append(vocab['<UNK>'])
        X_ind.append(X_temp)

    for line in y:
        y_temp = []
        for word in line.split():
            if word in vocab:
                y_temp.append(vocab[word])
            else:
                y_temp.append(vocab['<UNK>'])
        y_ind.append(y_temp)

    for lines in K:
        K_temp = []
        for line in lines:
            k_temp = []
            for word in line.split():
                if word in vocab:
                    k_temp.append(vocab[word])
                else:
                    k_temp.append(vocab['<UNK>'])
            K_temp.append(k_temp)
        K_ind.append(K_temp)

    return X_ind, y_ind, K_ind


class personaDataset(Dataset):
    def __init__(self, X, y, K, max_length):
