"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
from transformers import BertModel


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class PreTrainedBERT(nn.Module):
    """Baseline PreTrainedBERT model for SQuAD.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, device):
        super(PreTrainedBERT, self).__init__()
        self.device = device
        self.question_BERT = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True).to(self.device)
        self.context_BERT = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True).to(self.device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=560, nhead=8)
        self.qa_encoder = nn.TransformerEncoder(self.encoder_layer, 2)
        self.ffnn_nodes = 16
        # Converts tensor from size (b, max_len, 768) to (b, max_len, whatever size you choose)
        self.start_token_weights_1 = nn.Linear(768, self.ffnn_nodes)
        self.start_token_weights_2 = nn.Linear(self.ffnn_nodes, 1)
        self.end_token_weights_1 = nn.Linear(768, self.ffnn_nodes)
        self.end_token_weights_2 = nn.Linear(self.ffnn_nodes, 1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        # self.log_softmax = nn.Softmax(dim=1)

    def forward(self, questions_input_ids, questions_attn_mask, questions_token_type_ids, contexts_input_ids, contexts_attn_mask, contexts_token_type_ids):
        # print("Started forward pass")
        question_bert_output = self.question_BERT(questions_input_ids, questions_attn_mask)
        contexts_bert_output = self.context_BERT(contexts_input_ids, contexts_attn_mask)
        # pooled_output = bert_output['pooler_output']
        question_hidden_state = question_bert_output['hidden_states'][-2]
        contexts_hidden_state = contexts_bert_output['hidden_states'][-2]
        model_hidden_state = torch.cat((question_hidden_state, contexts_hidden_state), 1)
        model_hidden_state = model_hidden_state.permute(0,2,1)
        ## TODO: Maybe pass in the mask
        qa_encoded = self.qa_encoder(model_hidden_state)
        model_hidden_state = model_hidden_state.permute(0,2,1)
        start_ffnn_output = self.start_token_weights_2(self.start_token_weights_1(model_hidden_state)).squeeze()
        # start_ffnn_output = self.start_token_weights_1(hidden_state).squeeze()
        end_ffnn_output = self.end_token_weights_2(self.end_token_weights_1(model_hidden_state)).squeeze()
        # end_ffnn_output = self.end_token_weights_1(hidden_state).squeeze()
        # So that after passing through softmax, they result in zero probability
        start_ffnn_output = start_ffnn_output[:,-512:]
        end_ffnn_output = end_ffnn_output[:,-512:]
        # Makes output of softmax -inf which makes it imopssible for backprop
        # neg_inf = float('-inf')
        neg_inf = -9999
        # Masking unattended words
        start_ffnn_output[contexts_attn_mask == 0] = neg_inf
        end_ffnn_output[contexts_attn_mask == 0] = neg_inf
        # Masking words that are not context words before softmax
        # start_ffnn_output[token_type_ids != 0] = neg_inf
        # end_ffnn_output[token_type_ids != 0] = neg_inf
        # print(self.log_softmax(start_ffnn_output))
        log_p1, log_p2 = self.log_softmax(start_ffnn_output), self.log_softmax(end_ffnn_output)
        out = (torch.squeeze(log_p1), torch.squeeze(log_p2))
        # print(out[0][0])
        return out
