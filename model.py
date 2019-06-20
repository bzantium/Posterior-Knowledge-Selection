import math
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torchnlp.word_to_vector import GloVe
from utils import gumbel_softmax
import params


class Encoder(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hidden, n_layer, vocab=None):
        super(Encoder, self).__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        if vocab is None:
            self.embedding = nn.Embedding(n_vocab, n_embed)
        else:
            embedding = torch.Tensor(n_vocab, n_embed)
            vectors = GloVe()
            for word in vocab.stoi:
                if word in vectors.stoi:
                    embedding[vocab.stoi[word]] = vectors[word]
            self.embedding = nn.Embedding.from_pretrained(embedding)
            print("encoder embedding is initialized with Glove")
        self.gru = nn.GRU(input_size=n_embed, hidden_size=n_hidden,
                          num_layers=n_layer, bidirectional=True)

    def forward(self, X):
        '''
        :param X:
            Variable of shape (n_batch(B), seq_len(T)), which is utterance
        :return:
            GRU outputs in shape (T, B, n_hidden(H))
            last hidden state in shape (2(bi-directional)*n_layer(L), B, H)
            encoded utterance defined at paper in shape (B, 2*H)
        '''
        n_batch = X.size(0)
        inputs = self.embedding(X)
        inputs = inputs.transpose(0, 1)
        seq_lengths = torch.sum(X > 0, dim=-1)  # [n_batch]
        packed_inputs = rnn.pack_padded_sequence(inputs, seq_lengths, enforce_sorted=False)
        packed_outputs, hidden = self.gru(packed_inputs)  # hidden: [2*n_layer, n_batch, n_hidden]
        last_hidden = hidden.view(self.n_layer, 2, n_batch, self.n_hidden)
        f_hidden, b_hidden = last_hidden[-1]
        outputs, _ = rnn.pad_packed_sequence(packed_outputs)
        outputs = (outputs[:, :, :self.n_hidden] + outputs[:, :, self.n_hidden:])
        # outputs: [seq_len, n_batch, n_hidden]
        encoded = torch.cat((f_hidden, b_hidden), dim=1)  # encoded: [n_batch, 2*n_hidden]
        return outputs, hidden, encoded


class KnowledgeEncoder(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hidden, n_layer, vocab=None):
        super(KnowledgeEncoder, self).__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        if vocab is None:
            self.embedding = nn.Embedding(n_vocab, n_embed)
        else:
            embedding = torch.Tensor(n_vocab, n_embed)
            vectors = GloVe()
            for word in vocab.stoi:
                if word in vectors.stoi:
                    embedding[vocab.stoi[word]] = vectors[word]
            self.embedding = nn.Embedding.from_pretrained(embedding)
            print("Kencoder embedding is initialized with Glove")
        self.gru = nn.GRU(input_size=n_embed, hidden_size=n_hidden,
                          num_layers=n_layer, bidirectional=True)

    def forward(self, K):
        '''
        :param K:
             Variable of shape (B, n_knowledge(N), T) (knowledge)
             or (B, T), which is y (response)
        :return:
            encoded knowledge or encoded response in shape (B, N, 2*H)
        '''
        if len(K.shape) == 3:  # [n_batch, N, seq_len]
            n_batch = K.size(0)
            N = K.size(1)
            inputs = self.embedding(K)
            inputs = inputs.transpose(0, 1)  # [N, n_batch, seq_len, n_embed]
            encoded = torch.zeros(N, n_batch, 2*self.n_hidden)
            for i in range(N):
                k = inputs[i].transpose(0, 1)  # [n_batch, seq_len, n_embed]
                seq_lengths = torch.sum(K[:, i] > 0, dim=-1)
                packed_inputs = rnn.pack_padded_sequence(k, seq_lengths, enforce_sorted=False)
                _, hidden = self.gru(packed_inputs)  # hidden: [2*n_layer, n_batch, n_hidden]
                hidden = hidden.view(self.n_layer, 2, n_batch, self.n_hidden)
                f_hidden, b_hidden = hidden[-1]
                encoded[i] = torch.cat((f_hidden, b_hidden), dim=1)  # encoded: [n_batch, 2*n_hidden]
            return encoded.transpose(0, 1).cuda()  # [n_batch, N, 2*n_hidden]

        else:  # [n_batch, seq_len]
            y = K[:, 1:]
            n_batch = y.size(0)
            inputs = self.embedding(y)
            inputs = inputs.transpose(0, 1)  # [seq_len, n_batch, n_embed]
            seq_lengths = torch.sum(y > 0, dim=-1)  # [n_batch]
            packed_inputs = rnn.pack_padded_sequence(inputs, seq_lengths, enforce_sorted=False)
            _, hidden = self.gru(packed_inputs)  # hidden: [2*n_layer, n_batch, n_hidden]
            hidden = hidden.view(self.n_layer, 2, n_batch, self.n_hidden)
            f_hidden, b_hidden = hidden[-1]
            encoded = torch.cat((f_hidden, b_hidden), dim=1)  # encoded: [n_batch, 2*n_hidden]
            return encoded


class Manager(nn.Module):
    def __init__(self, n_hidden, n_vocab, temperature):
        super(Manager, self).__init__()
        self.n_hidden = n_hidden
        self.n_vocab = n_vocab
        self.temperature = temperature
        self.mlp = nn.Sequential(nn.Linear(4*n_hidden, 2*n_hidden))
        self.mlp_k = nn.Sequential(nn.Linear(2*n_hidden, n_vocab))

    def forward(self, x, y, K):
        '''
        :param x:
            encoded utterance in shape (B, 2*H)
        :param y:
            encoded response in shape (B, 2*H) (optional)
        :param K:
            encoded knowledge in shape (B, N, 2*H)
        :return:
            prior, posterior, selected knowledge, selected knowledge logits for BOW_loss
        '''
        if y is not None:
            prior = F.log_softmax(torch.bmm(x.unsqueeze(1), K.transpose(-1, -2)), dim=-1).squeeze(1)
            response = self.mlp(torch.cat((x, y), dim=-1))  # response: [n_batch, 2*n_hidden]
            K = K.transpose(-1, -2)  # K: [n_batch, 2*n_hidden, N]
            posterior_logits = torch.bmm(response.unsqueeze(1), K).squeeze(1)
            posterior = F.softmax(posterior_logits, dim=-1)
            k_idx = gumbel_softmax(posterior_logits, self.temperature)  # k_idx: [n_batch, N(one_hot)]
            k_i = torch.bmm(K, k_idx.unsqueeze(2)).squeeze(2)  # k_i: [n_batch, 2*n_hidden]
            k_logits = F.log_softmax(self.mlp_k(k_i), dim=-1)  # k_logits: [n_batch, n_vocab]
            return prior, posterior, k_i, k_logits  # prior: [n_batch, N], posterior: [n_batch, N]
        else:
            n_batch = K.size(0)
            k_i = torch.Tensor(n_batch, 2*self.n_hidden).cuda()
            prior = torch.bmm(x.unsqueeze(1), K.transpose(-1, -2)).squeeze(1)
            k_idx = prior.max(1)[1].unsqueeze(1)  # k_idx: [n_batch, 1]
            for i in range(n_batch):
                k_i[i] = K[i, k_idx[i]]
            return k_i


class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.n_hidden = n_hidden
        self.attn = nn.Linear(2 * n_hidden, n_hidden)
        self.v = nn.Parameter(torch.rand(n_hidden))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs, encoder_mask=None):  # hidden: [n_batch, n_hidden]
        seq_len = encoder_outputs.size(0)  # encoder_outputs: [seq_len, n_batch, n_hidden]
        h = hidden.repeat(seq_len, 1, 1).transpose(0, 1)  # [n_batch, seq_len, n_hidden]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [n_batch, seq_len, n_hidden]
        attn_weights = self.score(h, encoder_outputs, encoder_mask)  # [n_batch, 1, seq_len]
        return attn_weights

    def score(self, hidden, encoder_outputs, encoder_mask=None):
        # hidden: [n_batch, seq_len, n_hidden], encoder_outputs: [n_batch, seq_len, n_hidden]
        attn_scores = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=-1)))
        # attn_scores: [n_batch, seq_len, n_hidden]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [n_batch, 1, n_hidden]
        attn_scores = torch.bmm(v, attn_scores.transpose(1, 2))  # [n_batch, 1, seq_len]
        if encoder_mask is not None:
            attn_scores.masked_fill_(encoder_mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [n_batch, 1, seq_len]
        return attn_weights  # [n_batch, 1, seq_len]


class Decoder(nn.Module):  # Hierarchical Gated Fusion Unit
    def __init__(self, n_vocab, n_embed, n_hidden, n_layer, vocab=None):
        super(Decoder, self).__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        if vocab is None:
            self.embedding = nn.Embedding(n_vocab, n_embed)
        else:
            embedding = torch.Tensor(n_vocab, n_embed)
            vectors = GloVe()
            for word in vocab.stoi:
                if word in vectors.stoi:
                    embedding[vocab.stoi[word]] = vectors[word]
            self.embedding = nn.Embedding.from_pretrained(embedding)
            print("decoder embedding is initialized with Glove")
        self.attention = Attention(n_hidden)
        self.y_weight = nn.Linear(n_hidden, n_hidden)
        self.k_weight = nn.Linear(n_hidden, n_hidden)
        self.z_weight = nn.Linear(2 * n_hidden, n_hidden)
        self.y_gru = nn.GRU(n_embed + n_hidden, n_hidden, n_layer)
        self.k_gru = nn.GRU(3 * n_hidden, n_hidden, n_layer)
        self.out = nn.Linear(2 * n_hidden, n_vocab)

    def forward(self, input, k, hidden, encoder_outputs, encoder_mask=None):
        '''
        :param input:
            word_input for current time step, in shape (B)
        :param k:
            selected knowledge in shape (B, 2*H)
        :param hidden:
            last hidden state of the decoder, in shape (L, B, H)
        :param encoder_outputs:
            encoder outputs in shape (T, B, H)
        :param encoder_mask:
            encoder mask in shape (B, 1, T)
        :return:
            decoder output, next hidden state of the decoder, attention weights
        '''
        embedded = self.embedding(input).unsqueeze(0)  # [1, n_batch, n_embed]
        attn_weights = self.attention(hidden[-1], encoder_outputs, encoder_mask)  # [n_batch, 1, seq_len]
        context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))  # [n_batch, 1, n_hidden]
        context = context.transpose(0, 1)  # [1, n_batch, n_hidden]
        y_input = torch.cat((embedded, context), dim=-1)
        k_input = torch.cat((k.unsqueeze(0), context), dim=-1)
        y_output, y_hidden = self.y_gru(y_input, hidden)  # y_hidden: [n_layer, n_batch, n_hidden]
        k_output, k_hidden = self.k_gru(k_input, hidden)  # k_hidden: [n_layer, n_batch, n_hidden]
        t_hidden = torch.tanh(torch.cat((self.y_weight(y_hidden), self.k_weight(k_hidden)), dim=-1))
        # t_hidden: [n_layer, n_batch, 2*n_hidden]
        r = torch.sigmoid(self.z_weight(t_hidden))  # [n_layer, n_batch, n_hidden]
        hidden = torch.mul(r, y_hidden) + torch.mul(1-r, k_hidden) # [n_layer, n_batch, n_hidden]
        output = hidden[-1]  # [n_batch, n_hidden]
        context = context.squeeze(0)  # [n_batch, 2*n_hidden]
        output = self.out(torch.cat((output, context), dim=1))  # [n_batch, n_vocab]
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights
