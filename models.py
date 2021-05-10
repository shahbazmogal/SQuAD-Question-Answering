"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


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
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(PreTrainedBERT, self).__init__()
        self.BERT = BertModel.from_pretrained('bert-base-uncased')
        self.start_weights = nn.Linear(320, 320)
        self.end_weights = nn.Linear(320, 320)
        softmax = nn.Softmax(dim=1)

    def forward(self, b_contexts, b_questions):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        sequence_tuples = zip(b_contexts, b_questions)
        encoded_dict = tokenizer.batch_encode_plus(
                        sequence_tuples,                      # Context to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 320,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
        # print(encoded_dict.keys())
        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        bert_output = self.BERT(input_ids, attention_mask)
        last_hidden_state = bert_output['last_hidden_state']
        print(input_ids.shape)
        print(attention_mask.shape)
        print(last_hidden_state.shape)

        exit()
        # print(cw_idxs.shape) 
        out = (cw_idxs, qw_idxs)
        # c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        # q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        # c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        # q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        # c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        # q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        # att = self.att(c_enc, q_enc,
        #                c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        # mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        # out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
