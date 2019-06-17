# PostKS

#### Pytorch implementation of [Learning to Select Knowledge for Response Generation in Dialog Systems](https://arxiv.org/pdf/1902.04911.pdf)


<p align="center">
  <img src="https://github.com/bzantium/PostKS/blob/master/image/architecture.PNG">
</p>

<br><br>
## Requirement
- pytorch
- pytorch-nlp

<br><br>
## Run train
```
$ python train.py -pre_epoch 5 -n_epoch 15 -n_batch 128
```

<br><br>
## Run test
```
$ python test.py -n_batch 128
```

<br><br>
## model

- UtteranceEncoder
```
Encoder(
  (embedding): Embedding(20000, 300)
  (gru): GRU(300, 800, num_layers=2, bidirectional=True)
)
```

- KnowledgeEncoder
```
KnowledgeEncoder(
  (embedding): Embedding(20000, 300)
  (gru): GRU(300, 800, num_layers=2, bidirectional=True)
)
```

- KnowledgeManager
```
Manager(
  (mlp): Sequential(
    (0): Linear(in_features=3200, out_features=1600, bias=True)
  )
  (mlp_k): Sequential(
    (0): Linear(in_features=1600, out_features=20000, bias=True)
  )
)
```

- Decoder
```
Decoder(
  (embedding): Embedding(20000, 300)
  (attention): Attention(
    (attn): Linear(in_features=1600, out_features=800, bias=True)
  )
  (y_weight): Linear(in_features=800, out_features=800, bias=True)
  (k_weight): Linear(in_features=800, out_features=800, bias=True)
  (z_weight): Linear(in_features=1600, out_features=800, bias=True)
  (y_gru): GRU(1100, 800, num_layers=2)
  (k_gru): GRU(2400, 800, num_layers=2)
  (out): Linear(in_features=1600, out_features=20000, bias=True)
)
```
