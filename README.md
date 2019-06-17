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
If you run train, vocab.json and trained parameters will be saved. Then you can run test.

<br><br>
## Run test
```
$ python test.py -n_batch 128
```
