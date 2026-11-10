# coding: UTF-8
import torch
import torch.nn as nn
from asr.semantic_vad.models.bert_model.modeling import BertModel


class Bert(nn.Module):
    def __init__(self, bert_config):
        super(Bert, self).__init__()
        self.bert = BertModel(bert_config)
        self.fc = nn.Linear(bert_config.hidden_size, 1)

    def forward(self, context, mask):
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        logit = torch.sigmoid(out)
        return logit
