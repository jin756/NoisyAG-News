import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel


# https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
class TextAlbert(nn.Module):

  def __init__(self, model_config, bert_backbone, args):
    super(TextAlbert, self).__init__()
    self.num_classes = model_config['num_classes']
    model_name = "./models/albert-base-v2/"

    assert bert_backbone is None
    # 在这里加载预训练模型
    albertConfig = AlbertConfig.from_pretrained(model_name)

    self.Albert = AlbertModel.from_pretrained(model_name, config=albertConfig)
    for param in self.Albert.parameters():
      param.requires_grad = True
    self.drop = nn.Dropout(p=model_config['drop_rate'])
    self.classifier = nn.Linear(self.Albert.config.hidden_size, self.num_classes)


  def forward(self, input_ids, attention_mask):
      outputs = self.Albert(input_ids, attention_mask=attention_mask)
      sequence_output = outputs.last_hidden_state[:, 0, :]
      output = self.drop(sequence_output)
      logits = self.classifier(output)

      return {'logits': logits,'pooler_repr':sequence_output}


