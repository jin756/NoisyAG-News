import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


# https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
class TextGPT2(nn.Module):

  def __init__(self, model_config, bert_backbone, args):
    super(TextGPT2, self).__init__()
    self.num_classes = model_config['num_classes']
    model_name = "./models/GPT-2/"

    assert bert_backbone is None, 'we do not support training based on provided checkpoints yet'
    # 在这里加载预训练模型
    gpt2Config = GPT2Config.from_pretrained(model_name)

    self.gpt2 = GPT2Model.from_pretrained(model_name, config=gpt2Config)
    for param in self.gpt2.parameters():
      param.requires_grad = True
    self.drop = nn.Dropout(p=model_config['drop_rate'])
    self.classifier = nn.Linear(self.gpt2.config.n_embd, self.num_classes)


  def forward(self, input_ids, attention_mask):
      outputs = self.gpt2(input_ids, attention_mask=attention_mask)
      sequence_output = outputs.last_hidden_state[:, 0, :]
      output = self.drop(sequence_output)
      logits = self.classifier(output)

      return {'logits': logits}


