import os
import json
import torch
import params
import nltk
from utils import init_model, Vocabulary, knowledgeToIndex
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

    print("loading model...")
    encoder = Encoder(n_vocab, n_embed, n_hidden, n_layer).cuda()
    Kencoder = KnowledgeEncoder(n_vocab, n_embed, n_hidden, n_layer).cuda()
    manager = Manager(n_hidden, n_vocab, temperature).cuda()
    decoder = Decoder(n_vocab, n_embed, n_hidden, n_layer).cuda()

    encoder = init_model(encoder, restore=params.encoder_restore)
    Kencoder = init_model(Kencoder, restore=params.Kencoder_restore)
    manager = init_model(manager, restore=params.manager_restore)
    decoder = init_model(decoder, restore=params.decoder_restore)
    print("successfully loaded!\n")

    utterance = ""
    while True:
        if utterance == "exit":
            break
        k1 = input("Type first Knowledge: ").lower()
        k2 = input("Type second Knowledge: ").lower()
        k3 = input("Type third Knowledge: ").lower()

        K = [k1, k2, k3]
        K = knowledgeToIndex(K, vocab)
        K = Kencoder(K)
        print()

        while True:
            utterance = input("you: ").lower()
            if utterance == "change knowledge" or utterance == "exit":
                print()
                break

            X = []
            tokens = nltk.word_tokenize(utterance)
            for word in tokens:
                if word in vocab.stoi:
                    X.append(vocab.stoi[word])
                else:
                    X.append(vocab.stoi["<UNK>"])
            X = torch.LongTensor(X).unsqueeze(0).cuda()  # X: [1, x_seq_len]

            encoder_outputs, hidden, x = encoder(X)
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
                if idx == params.EOS:
                    break
                answer += vocab.itos[idx] + " "

            print("bot:", answer[:-1], "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
