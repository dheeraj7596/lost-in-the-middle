from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
from torch import nn
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
import warnings


class AlphaLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.alpha = nn.Parameter(torch.normal(mean=torch.tensor(2.0), std=torch.tensor(1.0)))

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        # hidden_states = residual + hidden_states
        residual = residual.to("cuda")
        hidden_states = hidden_states.to("cuda")
        norm_mult = torch.norm(residual + hidden_states, dim=-1).to("cuda")
        norm_mult = norm_mult.unsqueeze(-1).expand(-1, -1, residual.shape[-1])
        hidden_states = F.normalize(self.alpha * residual + hidden_states, dim=-1) * norm_mult

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # hidden_states = residual + hidden_states
        residual = residual.to("cuda")
        hidden_states = hidden_states.to("cuda")
        norm_mult = torch.norm(residual + hidden_states, dim=-1).to("cuda")
        norm_mult = norm_mult.unsqueeze(-1).expand(-1, -1, residual.shape[-1])
        hidden_states = F.normalize(self.alpha * residual + hidden_states, dim=-1) * norm_mult

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class AlphaLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AlphaLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class AlphaLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = AlphaLlamaModel(config)

