import random
import argparse
import torch
from torch import optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import params
from utils import init_model, save_model, \
    build_vocab, load_data, get_data_loader
from model import Encoder, KnowledgeEncoder, Decoder, Manager


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-pre_epochs', type=int, default=5,
                   help='number of epochs for pre_train')
    p.add_argument('-n_epoch', type=int, default=15,
                   help='number of epochs for train')
    p.add_argument('-n_batch', type=int, default=128,
                   help='number of batches for train')
    p.add_argument('-lr', type=float, default=5e-4,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    p.add_argument('-tfr', type=float, default=0.8,
                   help='teacher forcing ratio')
    p.add_argument('-train_path', type=str, default="",
                   help='path to train file')
    p.add_argument('-test_path', type=str, default="",
                   help='path to test file')
    p.add_argument('-restore', type=bool, default=False,
                   help='whether restore trained model')
    return p.parse_args()


def pre_train(model, optimizer, train_loader, args):
    encoder, Kencoder, manager, decoder = [*model]
    encoder.train(), Kencoder.train(), manager.train(), decoder.train()
    parameters = list(encoder.parameters()) + list(Kencoder.parameters()) + \
                 list(manager.parameters()) + list(decoder.parameters())
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)

    for epoch in range(args.pre_epochs):
        for step, (src_X, src_y, src_K, _) in enumerate(train_loader):
            src_X = src_X.cuda()
            src_y = src_y.cuda()
            src_K = src_K.cuda()

            optimizer.zero_grad()
            _, _, x = encoder(src_X)
            y = Kencoder(src_y)
            K = Kencoder(src_K)

            _, _, _, k_logits = manager(x, y, K)
            bow_loss = 0
            for i in range(1, src_y.size(1)):
                bow_loss += NLLLoss(k_logits, src_y[:, i])
            bow_loss.backward()
            clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            if step % 50 == 0:
                print("Epoch [%.1d/%.1d] Step [%.4d/%.4d]: loss=%.4f" % (epoch + 1, args.pre_epochs,
                                                                         step, len(train_loader),
                                                                         bow_loss.item()))


def train(model, optimizer, train_loader, args):
    encoder, Kencoder, manager, decoder = [*model]
    encoder.train(), Kencoder.train(), manager.train(), decoder.train()
    parameters = list(encoder.parameters()) + list(Kencoder.parameters()) + \
                 list(manager.parameters()) + list(decoder.parameters())
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)
    KLDLoss = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(args.n_epoch):
        for step, (src_X, src_y, src_K, tgt_y) in enumerate(train_loader):
            src_X = src_X.cuda()
            src_y = src_y.cuda()
            src_K = src_K.cuda()
            tgt_y = tgt_y.cuda()

            optimizer.zero_grad()
            encoder_outputs, hidden, x = encoder(src_X)
            y = Kencoder(src_y)
            K = Kencoder(src_K)
            prior, posterior, k_i, k_logits = manager(x, y, K)
            kldiv_loss = KLDLoss(torch.log(prior), posterior.detach())
            bow_loss = 0
            for i in range(1, src_y.size(1)):
                bow_loss += NLLLoss(k_logits, src_y[:, i])

            n_batch = src_X.size(0)
            max_len = tgt_y.size(1)
            n_vocab = decoder.n_vocab

            outputs = torch.zeros(max_len, n_batch, n_vocab).cuda()
            hidden = hidden[:decoder.n_layer]
            output = src_y[:, 0]  # [n_batch]
            for t in range(max_len):
                output, hidden, attn_weights = decoder(output, k_i, hidden, encoder_outputs)
                outputs[t] = output
                is_teacher = random.random() < args.tfr  # teacher forcing ratio
                top1 = output.data.max(1)[1]
                output = tgt_y[:, t] if is_teacher else top1

            nll_loss = NLLLoss(outputs.view(-1, n_vocab),
                               tgt_y.contiguous().view(-1))

            loss = kldiv_loss + nll_loss + bow_loss
            loss.backward()
            clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            # if (step + 1) % 50 == 0:
            print("Epoch [%.1d/%.1d] Step [%.4d/%.4d]: loss=%.4f" % (epoch + 1, args.n_epoch,
                                                                     step + 1, len(train_loader),
                                                                     loss.item()))


def main():
    args = parse_arguments()
    n_vocab = params.n_vocab
    n_layer = params.n_layer
    n_hidden = params.n_hidden
    n_embed = params.n_embed
    n_batch = args.n_batch
    temperature = 0.8
    train_path = "train_self_original_no_cands.txt"
    test_path = "valid_self_original_no_cands.txt"
    assert torch.cuda.is_available()

    print("loading_data...")
    vocab = build_vocab(train_path, n_vocab)
    train_X, train_y, train_K = load_data(train_path, vocab)
    test_X, test_y, test_K = load_data(test_path, vocab)
    train_loader = get_data_loader(train_X, train_y, train_K, n_batch)
    test_loader = get_data_loader(test_X, test_y, test_K, n_batch)
    print("successfully loaded")

    encoder = Encoder(n_vocab, n_embed, n_hidden, n_layer, vocab).cuda()
    Kencoder = KnowledgeEncoder(n_vocab, n_embed, n_hidden, n_layer, vocab).cuda()
    manager = Manager(n_hidden, n_vocab, temperature).cuda()
    decoder = Decoder(n_vocab, n_embed, n_hidden, n_layer, vocab).cuda()

    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
        Kencoder = torch.nn.DataParallel(Kencoder)
        manager = torch.nn.DataParallel(manager)
        decoder = torch.nn.DataParallel(decoder)

    if args.restore:
        encoder = init_model(encoder, restore=params.encoder_restore)
        Kencoder = init_model(Kencoder, restore=params.Kencoder_restore)
        manager = init_model(manager, restore=params.manager_restore)
        decoder = init_model(decoder, restore=params.decoder_restore)

    model = [encoder, Kencoder, manager, decoder]
    parameters = list(encoder.parameters()) + list(Kencoder.parameters()) + \
                 list(manager.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)

    # pre_train knowledge manager
    print("start pre-training")
    pre_train(model, optimizer, train_loader, args)
    print("start training")
    train(model, optimizer, train_loader, args)

    # save final model
    save_model(encoder, "PostKS-encoder.pt")
    save_model(Kencoder, "PostKS-Kencoder.pt")
    save_model(manager, "PostKS-manager.pt")
    save_model(decoder, "PostKS-decoder.pt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
