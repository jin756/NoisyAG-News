import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaModel


# https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
class TextRoBERTa(nn.Module):

  def __init__(self, model_config, bert_backbone, args):
    super(TextRoBERTa, self).__init__()
    self.num_classes = model_config['num_classes']

    assert bert_backbone is None, 'we do not support training based on provided checkpoints yet'
    self.bert = RobertaModel.from_pretrained("./models/" + args.model_name + "/")

   
    for param in self.bert.parameters():
      param.requires_grad = True

    self.drop = nn.Dropout(p=model_config['drop_rate'])
    self.out = nn.Linear(self.bert.config.hidden_size, self.num_classes)


  def forward(self, input_ids, attention_mask):
    bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    cls_repr = bert_out[0][:, 0, :]
    pooler_repr = bert_out['pooler_output']
    output = self.drop(pooler_repr)
    logits = self.out(output)
    # bert返回的logits是是没有经过softmax和argmax的

    return {'logits': logits, 'cls_repr': cls_repr, 'pooler_repr': pooler_repr}

