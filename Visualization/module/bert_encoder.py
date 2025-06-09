import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import CrossEntropyLoss
import ipdb

from .bert_config import BertConfig
from .bert_modeling import MultimodalBertMaskedLM


class MultiModalBertEncoder(nn.Module):
    def __init__(self):
        super(MultiModalBertEncoder, self).__init__()

        self.model = MultimodalBertMaskedLM(BertConfig())


    def forward(self, latent, gap_token, ids, attn_mask, token_type):

        output = self.model(latent, gap_token, ids, attn_mask, token_type)

        return output