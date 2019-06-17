import os
import json
import torch
import params
from utils import init_model, Vocabulary
from model import Encoder, KnowledgeEncoder, Decoder, Manager

def main():
    max_len = 50
    n_vocab = params.n_vocab
    n_layer = params.n_layer
    n_hidden = params.n_hidden
    n_embed = params.n_embed
    temperature = params.temperature
    assert torch.cuda.is_available()

    if os.path.exists("vocab.json"):
        vocab = Vocabulary()
        with open('vocab.json', 'r') as fp:
            vocab.stoi = json.load(fp)

        for key, value in vocab.stoi.items():
            vocab.itos.append(key)
    else:
        print("vocabulary doesn't exist!")
        return

    encoder = Encoder(n_vocab, n_embed, n_hidden, n_layer).cuda()
    Kencoder = KnowledgeEncoder(n_vocab, n_embed, n_hidden, n_layer).cuda()
    manager = Manager(n_hidden, n_vocab, temperature).cuda()
    decoder = Decoder(n_vocab, n_embed, n_hidden, n_layer).cuda()

    encoder = init_model(encoder, restore=params.encoder_restore)
    Kencoder = init_model(Kencoder, restore=params.Kencoder_restore)
    manager = init_model(manager, restore=params.manager_restore)
    decoder = init_model(decoder, restore=params.decoder_restore)

    k1 = input("Type first Knowledge: ")
    k2 = input("Type second Knowledge: ")
    k3 = input("Type third Knowledge: ")

    K1 = []
    K2 = []
    K3 = []

    for word in k1.split():
        if word in vocab.stoi:
            K1.append(vocab.stoi[word])
        else:
            K1.append(vocab.stoi["<UNK>"])

    for word in k2.split():
        if word in vocab.stoi:
            K2.append(vocab.stoi[word])
        else:
            K2.append(vocab.stoi["<UNK>"])

    for word in k3.split():
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
    print()

    while(True):
        utterance = input("you: ")
        X = []
        for word in utterance.split():
            if word in vocab.stoi:
                X.append(vocab.stoi[word])
            else:
                X.append(vocab.stoi["<UNK>"])
        X = torch.LongTensor(X).unsqueeze(0).cuda()  # X: [1, x_seq_len]

        encoder_outputs, hidden, x = encoder(X)
        K = Kencoder(K)
        k_i = manager(x, None, K)
        outputs = torch.zeros(max_len, 1, n_vocab).cuda()  # outputs: [max_len, 1, n_vocab]
        hidden = hidden[decoder.n_layer:]
        output = torch.LongTensor([params.SOS]).cuda()

        for t in range(max_len):
            output, hidden, attn_weights = decoder(output, k_i, hidden, encoder_outputs)
            outputs[t] = output
            output = output.data.max(1)[1]

        outputs = outputs.max(2)[1]

        answer = ""
        for idx in outputs:
            answer += vocab.itos[idx] + " "
            if idx == params.EOS:
                break

        print("bot:", answer[:-1])




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:

        print("[STOP]", e)