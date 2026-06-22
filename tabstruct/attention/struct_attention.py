
from tabstruct.attention.bias_utils import get_relative_relation_ids
from torch import nn
import torch
import math

from typing import Optional, Tuple


from transformers import BartConfig



class StructAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BartConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        self.encoding_structure_bias = config.encoding_structure_bias
        self.learnable_sparse_gate = bool(getattr(config, "learnable_sparse_gate", False))
        self.gate_epsilon = float(getattr(config, "gate_epsilon", 1e-6))
        self.latest_gate_loss = None
        self.latest_gate_stats = {}

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        
        if self.encoding_structure_bias == "B1":
            self.attention_bias_embeddings = nn.Embedding(13 + 1, 1)

        if self.learnable_sparse_gate:
            gate_hidden_dim = int(getattr(config, "gate_hidden_dim", 64))
            relation_dim = 9
            self.gate_query_proj = nn.Linear(embed_dim, num_heads * gate_hidden_dim, bias=False)
            self.gate_key_proj = nn.Linear(embed_dim, num_heads * gate_hidden_dim, bias=False)
            self.gate_relation_proj = nn.Linear(relation_dim, num_heads, bias=True)

            init_temperature = float(getattr(config, "gate_temperature", 1.0))
            init_temperature = max(init_temperature, self.gate_epsilon)
            if bool(getattr(config, "learnable_gate_temperature", False)):
                self.log_gate_temperature = nn.Parameter(torch.full((num_heads,), math.log(init_temperature)))
            else:
                self.register_buffer("log_gate_temperature", torch.full((num_heads,), math.log(init_temperature)))


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _build_relation_features(self, token_type: torch.LongTensor) -> torch.Tensor:
        segment_ids = token_type[:, :, 0]
        column_ids = token_type[:, :, 1]
        row_ids = token_type[:, :, 2]

        same_column = column_ids.unsqueeze(1) == column_ids.unsqueeze(2)
        same_row = row_ids.unsqueeze(1) == row_ids.unsqueeze(2)
        same_cell = same_row & same_column
        query_mask = segment_ids == 0
        header_mask = (row_ids == 0) & (segment_ids == 1)
        cell_mask = (row_ids != 0) & (segment_ids == 1)

        pair_features = [
            same_row,
            same_column,
            same_cell,
            query_mask.unsqueeze(2).expand_as(same_row),
            query_mask.unsqueeze(1).expand_as(same_row),
            header_mask.unsqueeze(2).expand_as(same_row),
            header_mask.unsqueeze(1).expand_as(same_row),
            cell_mask.unsqueeze(2).expand_as(same_row),
            cell_mask.unsqueeze(1).expand_as(same_row),
        ]
        return torch.stack(pair_features, dim=-1).to(dtype=torch.float32)

    def _compute_sparse_gate(
        self,
        hidden_states: torch.Tensor,
        token_type: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()
        gate_hidden_dim = self.gate_query_proj.out_features // self.num_heads

        query_gate = self.gate_query_proj(hidden_states).view(bsz, seq_len, self.num_heads, gate_hidden_dim)
        key_gate = self.gate_key_proj(hidden_states).view(bsz, seq_len, self.num_heads, gate_hidden_dim)
        query_gate = query_gate.permute(0, 2, 1, 3)
        key_gate = key_gate.permute(0, 2, 1, 3)

        relation_features = self._build_relation_features(token_type).to(device=hidden_states.device, dtype=hidden_states.dtype)
        relation_logits = self.gate_relation_proj(relation_features).permute(0, 3, 1, 2)
        content_logits = torch.matmul(query_gate, key_gate.transpose(-1, -2)) / math.sqrt(gate_hidden_dim)
        gate_logits = content_logits + relation_logits

        temperature = self.log_gate_temperature.exp().clamp_min(self.gate_epsilon).view(1, self.num_heads, 1, 1)
        gate_probs = torch.sigmoid(gate_logits / temperature)

        allowed_edges = torch.isfinite(attention_mask) & (attention_mask > torch.finfo(attention_mask.dtype).min / 2)
        allowed_edges = allowed_edges.expand(-1, self.num_heads, -1, -1)
        allowed_gate_probs = gate_probs.masked_select(allowed_edges)

        if allowed_gate_probs.numel() == 0:
            self.latest_gate_loss = gate_probs.new_zeros(())
            self.latest_gate_stats = {}
            return gate_probs

        sparsity_loss = allowed_gate_probs.mean()
        entropy = -(
            allowed_gate_probs * torch.log(allowed_gate_probs + self.gate_epsilon)
            + (1.0 - allowed_gate_probs) * torch.log(1.0 - allowed_gate_probs + self.gate_epsilon)
        ).mean()

        diversity_loss = gate_probs.new_zeros(())
        if self.num_heads > 1:
            head_masks = gate_probs.masked_fill(~allowed_edges, 0.0).flatten(start_dim=2)
            normed = nn.functional.normalize(head_masks, p=2, dim=-1, eps=self.gate_epsilon)
            cosine = torch.matmul(normed, normed.transpose(1, 2))
            off_diagonal = ~torch.eye(self.num_heads, dtype=torch.bool, device=gate_probs.device)
            diversity_loss = cosine[:, off_diagonal].mean()

        self.latest_gate_loss = (
            float(getattr(self.config, "sparsity_loss_weight", 0.0)) * sparsity_loss
            + float(getattr(self.config, "diversity_loss_weight", 0.0)) * diversity_loss
            + float(getattr(self.config, "entropy_loss_weight", 0.0)) * entropy
        )
        self.latest_gate_stats = {
            "sparsity": sparsity_loss.detach(),
            "entropy": entropy.detach(),
            "diversity": diversity_loss.detach(),
        }
        return gate_probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        token_type: torch.LongTensor = None,

    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

      

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)


        src_len = key_states.size(1)
        
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            

            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask

            if self.encoding_structure_bias == "B1":

                attention_bias_ids = get_relative_relation_ids(token_type, attention_mask)
                attention_bias = self.attention_bias_embeddings(attention_bias_ids).squeeze(-1).unsqueeze(1)
                attention_bias = attention_bias.expand(-1, self.num_heads, -1, -1)
                attn_weights = attn_weights + attention_bias

            if self.learnable_sparse_gate and not is_cross_attention:
                if token_type is None:
                    raise ValueError("token_type is required when learnable_sparse_gate is enabled.")
                gate_probs = self._compute_sparse_gate(hidden_states, token_type, attention_mask)
                attn_weights = attn_weights + torch.log(gate_probs + self.gate_epsilon)

            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
