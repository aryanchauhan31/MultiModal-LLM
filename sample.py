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

from torch.utils.data import DataLoader

import os

# Define the root path where you expect the unzipped files
root_path = '/content/'

# List all contents in the root directory
print("Root Directory Contents:")
print(os.listdir(root_path))

# Check if 'train2017_unzipped' exists
unzipped_dir = os.path.join(root_path, 'train2017_unzipped')
if os.path.exists(unzipped_dir):
    print("Unzipped Directory Contents:")
    print(os.listdir(unzipped_dir))
else:
    print(f"Directory not found: {unzipped_dir}")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
questions = ["What is in the image?"]
image_path = "example.jpg"
text_inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
image = Image.open(image_path).convert("RGB")
image_inputs = feature_extractor(images=image, return_tensors="pt")["pixel_values"]
print("Text Inputs:", text_inputs)
print("Image Inputs Shape:", image_inputs.shape)
model = CrossModalModel()
fused_outputs, task_output = model(text_inputs=text_inputs, image_inputs=image_inputs)
print("Fused Outputs Shape:", fused_outputs.shape)




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

from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer

class MSCOCODataset(Dataset):
    def __init__(self, annotations_path, images_dir, feature_extractor, tokenizer):
        with open(annotations_path, 'r') as f:
            self.coco_data = json.load(f)

        self.images_dir = images_dir  
        self.feature_extractor = feature_extractor  
        self.tokenizer = tokenizer 

    def __len__(self):
        return len(self.coco_data['annotations'])

    def __getitem__(self, idx):
        annotation = self.coco_data['annotations'][idx]
        caption = annotation['caption']
        image_id = annotation['image_id']
        image_path = f"{self.images_dir}/{str(image_id).zfill(12)}.jpg"
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            return None

        image_inputs = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"]
        text_inputs = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
        return {
            "image_inputs": image_inputs.squeeze(0),
            "text_inputs": {key: val.squeeze(0) for key, val in text_inputs.items()},
            "caption": caption,  # Include caption in the output
        }




annotations_path = './train2017_unzipped/annotations/captions_train2017.json'
images_dir = './train2017_unzipped/train2017'
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = MSCOCODataset(
    annotations_path=annotations_path,
    images_dir=images_dir,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

for batch in dataloader:
    print("Image Inputs Shape:", batch["image_inputs"].shape)  # [batch_size, 3, 224, 224]
    print("Text Inputs Shape:", batch["text_inputs"]['input_ids'].shape)  # [batch_size, seq_len]
    print("Captions:", batch["captions"])  # List of captions in the batch
    break




import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, AutoTokenizer

annotations_path = './train2017_unzipped/annotations/captions_train2017.json'
images_dir = './train2017_unzipped/train2017'

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

class MSCOCODataset(Dataset):
    def __init__(self, annotations_path, images_dir, feature_extractor, tokenizer):
        with open(annotations_path, 'r') as f:
            self.coco_data = json.load(f)
        self.images_dir = images_dir
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.coco_data['annotations'])

    def __getitem__(self, idx):
        annotation = self.coco_data['annotations'][idx]
        caption = annotation['caption']
        image_id = annotation['image_id']
        image_path = f"{self.images_dir}/{str(image_id).zfill(12)}.jpg"

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            return None

        image_inputs = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"]
        text_inputs = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True)

        return {
            "image_inputs": image_inputs.squeeze(0),
            "text_inputs": {key: val.squeeze(0) for key, val in text_inputs.items()},
            "caption": caption,
        }


def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Filter out None values
    image_inputs = torch.stack([item["image_inputs"] for item in batch])
    text_input_ids = [item["text_inputs"]["input_ids"] for item in batch]
    text_attention_mask = [item["text_inputs"]["attention_mask"] for item in batch]
    captions = [item["caption"] for item in batch]

    padded_input_ids = pad_sequence(text_input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence(text_attention_mask, batch_first=True, padding_value=0)

    return {
        "image_inputs": image_inputs,
        "text_inputs": {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
        },
        "captions": captions,
    }

dataset = MSCOCODataset(
    annotations_path=annotations_path,
    images_dir=images_dir,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

model = CrossModalModel().to('cuda')

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        image_inputs = batch["image_inputs"].to('cuda')
        text_inputs = {key: val.to('cuda') for key, val in batch["text_inputs"].items()}

        optimizer.zero_grad()

        fused_outputs, logits = model(text_inputs=text_inputs, image_inputs=image_inputs)

        labels = torch.zeros(logits.size(0), dtype=torch.long).to('cuda')
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")


