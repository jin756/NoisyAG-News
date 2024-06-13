import torch
import torch.nn as nn
from transformers import T5Config,T5Model


# https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
class TextT5(nn.Module):

  def __init__(self, model_config, bert_backbone, args):
    super(TextT5, self).__init__()
    self.num_classes = model_config['num_classes']
    model_name = "./models/t5-base/"

    assert bert_backbone is None
    # 在这里加载预训练模型
    t5Config = T5Config.from_pretrained(model_name)

    self.t5 = T5Model.from_pretrained(model_name, config=t5Config)
    for param in self.t5.parameters():
      param.requires_grad = True
    self.drop = nn.Dropout(p=model_config['drop_rate'])
    self.classifier = nn.Linear(self.t5.config.d_model, self.num_classes)


  def forward(self, input_ids, attention_mask):
      outputs = self.t5(input_ids, attention_mask=attention_mask)
      sequence_output = outputs.last_hidden_state[:, 0, :]
      output = self.drop(sequence_output)
      logits = self.classifier(output)

      return {'logits': logits,'pooler_repr':sequence_output}


