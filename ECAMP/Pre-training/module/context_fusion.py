import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention, BertIntermediate, BertOutput, BertSelfOutput
from transformers.modeling_utils import apply_chunking_to_forward


class ECAMPFusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.cross_self_attention = BertSelfAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.gap_mlp = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_layer = BertSelfOutput(config)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            gap_token,
            attention_mask=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = None  # past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        cross_self_outputs = self.cross_self_attention(
            attention_output,
            attention_mask,
            None,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            output_attentions,
        )
        gap_token = self.gap_mlp(gap_token)
        cross_self_outputs = cross_self_outputs[0] + gap_token
        cross_attention_outputs = self.out_layer(cross_self_outputs, attention_output)

        attention_output = cross_attention_outputs
        # outputs = outputs + cross_attention_outputs[1:]
        # outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output