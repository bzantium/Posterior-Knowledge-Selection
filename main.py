import random
import argparse
import torch
from torch import optim
import torch.nn as nn
from model import Encoder, Decoder, Manager


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-pre_epochs', type=int, default=5,
                   help='number of epochs for train')
    p.add_argument('-epochs', type=int, default=15,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-tfr', type=float, default=0.8,
                   help='teacher forcing ratio')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    return p.parse_args()


def main():
    args = parse_arguments()
    n_layer = 2
    n_vocab = 20000
    n_hidden = 800
    n_embed = 300
    n_batch = 128
    assert torch.cuda.is_available()

    encoder = Encoder(n_vocab, n_embed, n_hidden, n_layer).cuda()
    manager = Manager(n_hidden).cuda()
    decoder = Decoder(n_vocab, n_embed, n_hidden, n_layer).cuda()
    parameters = list(encoder.parameters()) + list(manager.parameters()) \
                 + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)

    KLDivloss = nn.KLDivLoss(reduction='batchmean')
    NLLloss = nn.NLLLoss()
    for epoch in range(args.pre_epochs):
        optimizer.zero_grad()

        _, _, x = encoder(src.X)
        _, _, y = encoder(tgt)
        _, _, K = encoder(src.K)

        _, _, _, k_logits = manager(x, y, K)

        bow_loss = NLLloss(k_logits, tgt)
        bow_loss.backward()
        optimizer.step()

    for epoch in range(args.epochs):
        optimizer.zero_grad()

        encoder_output, hidden, x = encoder(src.X)
        _, _, y = encoder(tgt)
        _, _, K = encoder(src.K)

        prior, posterior, k_i, k_logits = manager(x, y, K)
        kldiv_loss = KLDivloss(torch.log(prior), posterior.detach())
        bow_loss = NLLloss(k_logits, tgt)

        n_batch = src.size(0)
        max_len = tgt.size(1)

        outputs = torch.zeros(max_len, n_batch, n_vocab)
        hidden = hidden[:decoder.n_layer]
        outputs[0] = tgt.data[:, 0, :]
        for t in range(1, max_len):
            output, hidden, attn_weights = decoder(output, k, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < args.tfr  # teacher forcing ratio
            top1 = output.data.max(1)[1]
            output = (tgt.data[t] if is_teacher else top1)

        nll_loss = NLLloss(outputs[1:].view(-1, n_vocab),
                           tgt[1:].contiguous().view(-1),
                           ignore_index = 0)

        loss = kldiv_loss + nll_loss + bow_loss
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

