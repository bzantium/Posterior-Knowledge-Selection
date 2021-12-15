import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import params
from collections import Counter
import pickle
import nltk
import gzip
import json

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


def init_model(net, restore=None):

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        print("Restore model from: {}".format(os.path.abspath(restore)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(), filename)
    print("save pretrained model to: {}".format(filename))


def save_models(model, filenames):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    for i in range(len(model)):
        net = model[i]
        filename = filenames[i]
        torch.save(net.state_dict(), filename)
        print("save pretrained model to: {}".format(filename))


def pickle_glove_vectors():
    glove_vectors = {}
    print("Start loading tensors from text file")
    with open("gloves/glove.840B.300d.txt", "r") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            try:
                glove_vectors[word] = torch.FloatTensor(np.array(split_line[1:], dtype=np.float32))
            except:
                pass
    print("Start to pickle glove vectors...")
    with gzip.open("gloves/glove_vectors.pkl", "wb") as f:
        pickle.dump(glove_vectors, f)
    print("Tensors are saved to gloves/glove_vectors.pkl")


def load_glove_vectors():
    print("Loading glove vectors")
    if os.path.exists("gloves/glove_vectors.pkl"):
        with gzip.open("gloves/glove_vectors.pkl", "rb") as f:
            glove_vectors = pickle.load(f)
    print(f"{len(glove_vectors)} words loaded!")
    return glove_vectors


def build_vocab(path, n_vocab):
    if os.path.exists("vocab.json"):
        print("Load vocab from vocab.json")
        vocab = Vocabulary()
        with open("vocab.json", "r") as f:
            vocab.stoi = json.load(f)
        for key in vocab.stoi.keys():
            vocab.itos.append(key)
    else:
        print("Build vocab...")
        with open(path, errors="ignore") as file:
            word_counter = Counter()
            vocab = Vocabulary()
            # vocab = dict()
            # reverse_vocab = dict()
            vocab.stoi['<PAD>'] = params.PAD
            vocab.stoi['<UNK>'] = params.UNK
            vocab.stoi['<SOS>'] = params.SOS
            vocab.stoi['<EOS>'] = params.EOS

            initial_vocab_size = len(vocab.stoi)
            vocab_idx = initial_vocab_size

            for line in file:
                dialog_id = line.split()[0]
                if dialog_id == "1":
                    count = 0

                if "your persona:" in line:
                    if count == 3:
                        continue
                    k_line = line.split("persona:")[1].strip("\n").lower()
                    tokens = nltk.word_tokenize(k_line)
                    count += 1

                    for word in tokens:
                        if word in vocab.itos:
                            word_counter[word] += 1
                        else:
                            word_counter[word] = 1

                elif "__SILENCE__" not in line:
                    X_line = " ".join(line.split("\t")[0].split()[1:]).lower()
                    tokens = nltk.word_tokenize(X_line)

                    for word in tokens:
                        if word in vocab.itos:
                            word_counter[word] += 1
                        else:
                            word_counter[word] = 1

                    y_line = line.split("\t")[1].strip("\n").lower()
                    tokens = nltk.word_tokenize(y_line)

                    for word in tokens:
                        if word in vocab.itos:
                            word_counter[word] += 1
                        else:
                            word_counter[word] = 1

            for key, _ in word_counter.most_common(n_vocab - initial_vocab_size):
                vocab.stoi[key] = vocab_idx
                vocab_idx += 1

            for key in vocab.stoi.keys():
                vocab.itos.append(key)

        with open('vocab.json', 'w') as fp:
            json.dump(vocab.stoi, fp)

    return vocab


def load_data(path, vocab):
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
                k_line = line.split("persona:")[1].strip("\n").lower()
                k.append(k_line)

            elif "__SILENCE__" not in line:
                K.append(k)
                X_line = " ".join(line.split("\t")[0].split()[1:]).lower()
                y_line = line.split("\t")[1].strip("\n").lower()
                X.append(X_line)
                y.append(y_line)

    X_ind = []
    y_ind = []
    K_ind = []

    for line in X:
        X_temp = []
        tokens = nltk.word_tokenize(line)
        for word in tokens:
            if word in vocab.stoi:
                X_temp.append(vocab.stoi[word])
            else:
                X_temp.append(vocab.stoi['<UNK>'])
        X_ind.append(X_temp)

    for line in y:
        y_temp = []
        tokens = nltk.word_tokenize(line)
        for word in tokens:
            if word in vocab.stoi:
                y_temp.append(vocab.stoi[word])
            else:
                y_temp.append(vocab.stoi['<UNK>'])
        y_ind.append(y_temp)

    for lines in K:
        K_temp = []
        for line in lines:
            k_temp = []
            tokens = nltk.word_tokenize(line)
            for word in tokens:
                if word in vocab.stoi:
                    k_temp.append(vocab.stoi[word])
                else:
                    k_temp.append(vocab.stoi['<UNK>'])
            K_temp.append(k_temp)
        K_ind.append(K_temp)

    return X_ind, y_ind, K_ind


def get_data_loader(X, y, K, n_batch):
    dataset = PersonaDataset(X, y, K)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=n_batch,
        shuffle=True
    )
    return data_loader


class Vocabulary:
    def __init__(self):
        self.itos = list()
        self.stoi = dict()


class PersonaDataset(Dataset):
    def __init__(self, X, y, K):
        X_len = max([len(line) for line in X])
        y_len = max([len(line) for line in y])
        k_len = 0
        for lines in K:
            for line in lines:
                if k_len < len(line):
                    k_len = len(line)

        src_X = list()
        src_y = list()
        src_K = list()
        tgt_y = list()

        for line in X:
            line.extend([params.PAD] * (X_len - len(line)))
            src_X.append(line)

        for line in y:
            src_line = line[:]
            tgt_line = line[:]
            src_line.insert(0, params.SOS)
            tgt_line.append(params.EOS)
            src_line.extend([params.PAD] * (y_len - len(src_line) + 1))
            tgt_line.extend([params.PAD] * (y_len - len(tgt_line) + 1))
            src_y.append(src_line)
            tgt_y.append(tgt_line)

        for lines in K:
            src_k = list()
            for line in lines:
                line.extend([params.PAD] * (k_len - len(line)))
                src_k.append(line)
            src_K.append(src_k)

        self.src_X = torch.LongTensor(src_X)
        self.src_y = torch.LongTensor(src_y)
        self.src_K = torch.LongTensor(src_K)
        self.tgt_y = torch.LongTensor(tgt_y)
        self.dataset_size = len(self.src_X)

    def __getitem__(self, index):
        src_X = self.src_X[index]
        src_y = self.src_y[index]
        tgt_y = self.tgt_y[index]
        src_K = self.src_K[index]
        return src_X, src_y, src_K, tgt_y

    def __len__(self):
        return self.dataset_size


def knowledgeToIndex(K, vocab):
    k1, k2, k3 = K
    K1 = []
    K2 = []
    K3 = []

    tokens = nltk.word_tokenize(k1)
    for word in tokens:
        if word in vocab.stoi:
            K1.append(vocab.stoi[word])
        else:
            K1.append(vocab.stoi["<UNK>"])

    tokens = nltk.word_tokenize(k2)
    for word in tokens:
        if word in vocab.stoi:
            K2.append(vocab.stoi[word])
        else:
            K2.append(vocab.stoi["<UNK>"])

    tokens = nltk.word_tokenize(k3)
    for word in tokens:
        if word in vocab.stoi:
            K3.append(vocab.stoi[word])
        else:
            K3.append(vocab.stoi["<UNK>"])

    K = [K1, K2, K3]
    seq_len = max([len(k) for k in K])

    K1.extend([0] * (seq_len - len(K1)))
    K2.extend([0] * (seq_len - len(K2)))
    K3.extend([0] * (seq_len - len(K3)))

    K1 = torch.LongTensor(K1).unsqueeze(0)
    K2 = torch.LongTensor(K2).unsqueeze(0)
    K3 = torch.LongTensor(K3).unsqueeze(0)
    K = torch.cat((K1, K2, K3), dim=0).unsqueeze(0).cuda()  # K: [1, 3, seq_len]
    return K


if __name__ == "__main__":
    pickle_glove_vectors()
