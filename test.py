import argparse
import json
import os

import torch
import torch.nn as nn

import params
from model import Decoder, Encoder, KnowledgeEncoder, Manager
from utils import (Vocabulary, build_vocab, get_data_loader, init_model,
                   load_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    p = argparse.ArgumentParser(description="Hyperparams")
    p.add_argument("-n_batch", type=int, default=128, help="number of epochs for train")
    return p.parse_args()


def evaluate(model, test_loader):
    encoder, Kencoder, manager, decoder = [*model]
    encoder.eval(), Kencoder.eval(), manager.eval(), decoder.eval()
    NLLLoss = nn.NLLLoss(reduction="mean", ignore_index=params.PAD)
    total_loss = 0

    for step, (src_X, _, src_K, tgt_y) in enumerate(test_loader):
        src_X = src_X.to(device)
        src_K = src_K.to(device)
        tgt_y = tgt_y.to(device)

        encoder_outputs, hidden, x = encoder(src_X)
        encoder_mask = (src_X == 0).unsqueeze(1).byte()
        K = Kencoder(src_K)
        k_i = manager(x, None, K)
        n_batch = src_X.size(0)
        max_len = tgt_y.size(1)
        n_vocab = params.n_vocab

        outputs = torch.zeros(max_len, n_batch, n_vocab).to(device)
        hidden = hidden[params.n_layer :]
        output = torch.LongTensor([params.SOS] * n_batch).to(device)  # [n_batch]
        for t in range(max_len):
            output, hidden, attn_weights = decoder(
                output, k_i, hidden, encoder_outputs, encoder_mask
            )
            outputs[t] = output
            output = output.data.max(1)[1]

        outputs = outputs.transpose(0, 1).contiguous()
        loss = NLLLoss(outputs.view(-1, n_vocab), tgt_y.contiguous().view(-1))
        total_loss += loss.item()
    total_loss /= len(test_loader)
    print("nll_loss=%.4f" % (total_loss))


def main():
    args = parse_arguments()
    n_vocab = params.n_vocab
    n_layer = params.n_layer
    n_hidden = params.n_hidden
    n_embed = params.n_embed
    n_batch = args.n_batch
    temperature = params.temperature
    test_path = params.test_path

    if os.path.exists("vocab.json"):
        vocab = Vocabulary()
        with open("vocab.json", "r") as f:
            vocab.stoi = json.load(f)

        for key in vocab.stoi.items():
            vocab.itos.append(key)
    else:
        print("vocabulary doesn't exist!")
        return

    print("loading_data...")
    test_X, test_y, test_K = load_data(test_path, vocab)
    test_loader = get_data_loader(test_X, test_y, test_K, n_batch)
    print("successfully loaded")

    encoder = Encoder(n_vocab, n_embed, n_hidden, n_layer).to(device)
    Kencoder = KnowledgeEncoder(n_vocab, n_embed, n_hidden, n_layer).to(device)
    manager = Manager(n_hidden, n_vocab, temperature).to(device)
    decoder = Decoder(n_vocab, n_embed, n_hidden, n_layer).to(device)

    encoder = init_model(encoder, restore=params.encoder_restore)
    Kencoder = init_model(Kencoder, restore=params.Kencoder_restore)
    manager = init_model(manager, restore=params.manager_restore)
    decoder = init_model(decoder, restore=params.decoder_restore)

    model = [encoder, Kencoder, manager, decoder]
    print("start evaluating")
    evaluate(model, test_loader)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
