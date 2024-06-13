import torch
import torch.nn as nn
from transformers import BartConfig, BartModel


# https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
class TextBART(nn.Module):

  def __init__(self, model_config, bert_backbone, args):
    super(TextBART, self).__init__()
    self.num_classes = model_config['num_classes']
    model_name = "./models/bart-base/"

    assert bert_backbone is None, 'we do not support training based on provided checkpoints yet'
    # 在这里加载预训练模型
    bartConfig = BartConfig.from_pretrained(model_name)

    self.xlnet = BartModel.from_pretrained(model_name, config=bartConfig)
    for param in self.xlnet.parameters():
      param.requires_grad = True
    self.drop = nn.Dropout(p=model_config['drop_rate'])
    self.classifier = nn.Linear(self.xlnet.config.d_model, self.num_classes)


  def forward(self, input_ids, attention_mask):
      outputs = self.xlnet(input_ids, attention_mask=attention_mask)
      sequence_output = outputs.last_hidden_state[:, 0, :]
      output = self.drop(sequence_output)
      logits = self.classifier(output)

      return {'logits': logits,'pooler_repr':sequence_output}


