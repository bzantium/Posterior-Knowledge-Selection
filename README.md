## PostKS (Posterior Knowledge Selection)

#### Pytorch implementation of [Learning to Select Knowledge for Response Generation in Dialog Systems](https://arxiv.org/pdf/1902.04911.pdf)
For decoder, I apply Hierarchical Gated Fusion Unit (HGFU) [[Yao et al. 2017](https://www.aclweb.org/anthology/D17-1233)] and I only use three number of knowledges for the sake of code simplicity.

<p align="center">
  <img src="https://github.com/bzantium/PostKS/blob/master/image/architecture.PNG">
</p>

<br><br>
## Requirement
- pytorch
- pytorch-nlp
- nltk
- nltk.download('punkt')

<br><br>
## Train model
#### If you run train, vocab.json and trained parameters will be saved. Then you can play demo.
```
$ python train.py -pre_epoch 5 -n_epoch 15 -n_batch 128
```

<br><br>
## Play demo
```
$ python demo.py
```
#### You need to type three knowledges and utterance. Then bot will reply!
```
# example
Type first Knowledge: i'm very athletic.
Type second Knowledge: i wear contacts.
Type third Knowledge: i have brown hair.

you: hi ! i work as a gourmet cook .
bot(response): i don't like carrots . i throw them away . # reponse can change based on training.
```
- If you type "change knowledge" at (you), you can retype three knowledges.
- If you type "exit" at (you), you can terminate demo.


<br><br>
## DataSet
- I only use "self_original_no_cands" in Persona-chat released by ParlAI
