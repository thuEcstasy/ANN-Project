from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
import jittor as jt
import torch as th
import jittor.nn as nn

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)


class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
    ):
        super().__init__()
        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with jt.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size, name='小 1'),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size, name='小 3'))
        
        if init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", jt.broadcast(jt.arange(config.max_position_embeddings), shape=(1, config.max_position_embeddings)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        else:
            assert False, "invalid type of init_pretrained"
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        
        if self.logits_mode == 1:
            print('mode == 1')
            return self.lm_head(hidden_repr)
        else:
            raise NotImplementedError


    def execute(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        emb_inputs = emb_inputs.numpy()
        emb_inputs = th.from_numpy(emb_inputs)
        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state.detach()
        
        input_trans_hidden_states = jt.from_torch(input_trans_hidden_states)
        
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.to(x.dtype)
        return h