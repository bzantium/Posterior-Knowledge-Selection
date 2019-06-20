PAD = 0
UNK = 1
SOS = 2
EOS = 3

n_vocab = 20000
n_layer = 2
n_hidden = 800
n_embed = 300
temperature = 0.8

train_path = "data/train_self_original_no_cands.txt"
test_path = "data/valid_self_original_no_cands.txt"

model_root = "snapshots"
encoder_restore = "snapshots/PostKS-encoder.pt"
Kencoder_restore = "snapshots/PostKS-Kencoder.pt"
manager_restore = "snapshots/PostKS-manager.pt"
decoder_restore = "snapshots/PostKS-decoder.pt"
all_restore=[encoder_restore, Kencoder_restore, manager_restore, decoder_restore]

