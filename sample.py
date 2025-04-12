!wget 'http://images.cocodataset.org/zips/train2017.zip'
!wget 'http://images.cocodataset.org/zips/test2017.zip'
!wget 'http://images.cocodataset.org/zips/val2017.zip'

import zipfile
import os

# zip_file_path = '/content/train2017.zip'
# destination_folder = './train2017_unzipped/'

zip_file_path = '/content/val2017.zip'
destination_folder = './val2017_unzipped/'

# zip_file_path = '/content/test2017.zip'
# destination_folder = './test2017_unzipped/'


if not os.path.exists(zip_file_path):
  print('File not Found')

else:
  with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)


from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import torch

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')

def preprocess_image(image_path):
  image = Image.open(image_path).convert("RGB")
  inputs = feature_extractor(images = image, return_tensors = 'pt')
  with torch.no_grad():
    outputs = vit_model(**inputs)
  return outputs.last_hidden_state


squad_dataset = load_dataset("squad_v2")
print(squad_dataset)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_squad(example):
  return tokenizer(example['question'], example['context'], truncation = True, padding = 'max_length', return_tensors = 'pt')

squad_dataset = squad_dataset.map(preprocess_squad, batched = True)


import torch
import torch.nn as nn

class CrossAttention(nn.Module):
  def __init__(self, hidden_dim):
    super().__init__()
    self.query_projection = nn.Linear(hidden_dim, hidden_dim)
    self.key_projection = nn.Linear(hidden_dim, hidden_dim)
    self.value_projection = nn.Linear(hidden_dim, hidden_dim)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, query, key, value):
    query = self.query_projection(query)
    key = self.key_projection(key)
    value = self.value_projection(value)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
    attention_weights = self.softmax(attention_scores)
    attended_values = torch.matmul(attention_weights, value)
    return attended_values
