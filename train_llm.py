import os
import math
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import wandb
from transformers import AutoTokenizer, ViTImageProcessor, get_cosine_schedule_with_warmup
import random 

os.environ["WANDB_API_KEY"] = "8c9400488ca055df967dcf50beeda322adfff0f0"
# Dataset class for MS COCO captions
class MSCOCODataset(Dataset):
    def __init__(self, annotations_path, images_dir, image_processor, tokenizer, reduce_fraction=0.55):
        # Load MS COCO annotations from JSON
        with open(annotations_path, 'r') as f:
            self.coco_data = json.load(f)
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        # Use and reduce the list of annotations
        annotations = self.coco_data['annotations']
        if reduce_fraction < 1.0:
            random.seed(42)
            annotations = random.sample(annotations, int(len(annotations) * reduce_fraction))
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        caption = ann['caption']
        image_id = ann['image_id']
        image_path = os.path.join(self.images_dir, f"{str(image_id).zfill(12)}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return None
        image_tensor = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        text_inputs = self.tokenizer(caption, return_tensors="pt", padding=False, truncation=True)
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        return {
            "image_inputs": image_tensor,
            "text_inputs": text_inputs,
            "caption": caption,
            "image_id": image_id
        }
# Collate function to handle variable-length captions and filter missing data
def collate_fn(batch):
    # Remove any None entries (e.g., missing images)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    # Stack all image tensors (they have the same shape after processing)
    image_inputs = torch.stack([item["image_inputs"] for item in batch], dim=0)
    # Pad tokenized captions to the same length
    input_ids_list = [item["text_inputs"]["input_ids"] for item in batch]
    attention_mask_list = [item["text_inputs"]["attention_mask"] for item in batch]
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    captions = [item["caption"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    # Return tuple of tensors (and captions) for the batch
    return image_inputs, padded_input_ids, padded_attention_mask, captions

def compute_contrastive_loss(logits, temperature=0.07):
    # Symmetric cross-entropy loss for aligned image-text pairs
    targets = torch.arange(logits.size(0)).to(logits.device)
    logits = logits / temperature  # apply temperature scaling
    loss_i2t = F.cross_entropy(logits, targets)       # image-to-text loss
    loss_t2i = F.cross_entropy(logits.T, targets)     # text-to-image loss
    return (loss_i2t + loss_t2i) / 2

def compute_recall(logits, topk=(1, 5)):
    # Compute Recall@K for image-to-text retrieval given similarity logits
    targets = torch.arange(logits.size(0)).to(logits.device)
    recall_scores = {}
    for k in topk:
        _, indices = logits.topk(k, dim=1)  # top-k indices for each image
        # Calculate fraction of correct matches (diagonal) in top-k
        correct = (indices == targets.unsqueeze(1)).float().sum().item()
        recall_scores[f"recall@{k}"] = correct / targets.size(0)
    return recall_scores

# Model definition: ViT image encoder, BERT text encoder, projection layers, and temperature parameter
class CrossModalModel(nn.Module):
    def __init__(self, image_model_name="google/vit-base-patch16-224", text_model_name="bert-base-uncased", embed_dim=256):
        super(CrossModalModel, self).__init__()
        from transformers import ViTModel, BertModel
        # Load pretrained vision and text encoders
        self.image_encoder = ViTModel.from_pretrained(image_model_name)
        self.text_encoder  = BertModel.from_pretrained(text_model_name)
        # Linear projection layers to a shared embedding dimension
        self.image_proj = nn.Linear(self.image_encoder.config.hidden_size, embed_dim)
        self.text_proj  = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        # Learnable logit scaling (temperature) parameter, initialized as log(1/0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07))
    def forward(self, image_inputs, text_inputs):
        # Encode image and text inputs
        image_outputs = self.image_encoder(pixel_values=image_inputs)
        text_outputs  = self.text_encoder(**text_inputs)
        # Use the [CLS] token embedding from each encoder output
        image_cls = image_outputs.last_hidden_state[:, 0, :]
        text_cls  = text_outputs.last_hidden_state[:, 0, :]
        # Project to the shared embedding space
        image_embeds = F.normalize(self.image_proj(image_cls), p=2, dim=-1)
        text_embeds  = F.normalize(self.text_proj(text_cls), p=2, dim=-1)
        # Compute cosine similarity logits scaled by the learnable temperature
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits = logit_scale * (image_embeds @ text_embeds.T)
        return logits

def main():
    # Initialize distributed training environment (NCCL backend)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Paths for MS COCO 2017 datasets (adjust if needed)
    train_annotations = "/scratch/ac11274/project/annotations/captions_train2017.json"
    train_images_dir  = "/scratch/ac11274/project/train2017"
    val_annotations   = "/scratch/ac11274/project/annotations/captions_val2017.json"
    val_images_dir    = "/scratch/ac11274/project/val2017"
    # Initialize feature extractor and tokenizer from HuggingFace
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Create datasets and distributed samplers
    train_dataset = MSCOCODataset(train_annotations, train_images_dir, image_processor, tokenizer)
    val_dataset   = MSCOCODataset(val_annotations,   val_images_dir,   image_processor, tokenizer)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)
    # Data loaders (with our fixed collate_fn)
    # train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler,
    #                           collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn)


    if len(train_loader) == 0:
        raise RuntimeError("Train loader is empty. Please check MSCOCO paths and data integrity.")
    # val_loader   = DataLoader(val_dataset,   batch_size=32, sampler=val_sampler,
    #                           collate_fn=collate_fn, num_workers=4, pin_memory=True)
    # # Initialize model and wrap with DDP
    model = CrossModalModel().to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # Optimizer and learning rate scheduler (cosine schedule with warmup)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 5  # 7 epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    # Optionally initialize W&B on rank 0
    if rank == 0:
        wandb.init(project="clip_training", name="CLIP_like_MSCOCO", config={
            "status": "initialized",
            "status": "initialized",
            "learning_rate": 2e-5, "epochs": 5, "batch_size": 64
        })
    # Training loop
    for epoch in range(7):
        model.train()
        train_sampler.set_epoch(epoch)  # shuffle differently each epoch for DistributedSampler
        skipped_batches = 0
        for i, batch in enumerate(train_loader):
            if batch is None:
                # Skip empty batch (e.g., if all images in this batch were missing)
                if rank == 0:
                    print(f"[rank{rank}] Skipped empty batch {i}")
                    wandb.log({"skipped_batch": i})
                    wandb.log({"skipped_batch": i})
                continue
            image_inputs, input_ids, attention_mask, _ = batch  # captions are not used in training
            # Move data to this process's GPU
            image_inputs = image_inputs.to(device, non_blocking=True)
            input_ids    = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            optimizer.zero_grad()
            # Forward pass: model returns similarity logits
            logits = model(image_inputs=image_inputs, text_inputs={"input_ids": input_ids, "attention_mask": attention_mask})
            loss = compute_contrastive_loss(logits, temperature=1.0)  # use temperature=1.0 because model already applies its learnable temperature
            loss.backward()
            nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            # Log training loss periodically on rank 0
            if rank == 0 and i % 10 == 0:
                wandb.log({"train_loss": loss.item(), "epoch": epoch})
        # Validation loop (evaluation on validation set)
        model.eval()
        val_losses = []
        all_metrics = []
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                if batch is None:
                    if rank == 0:
                        print(f"[rank{rank}] Skipped empty val batch {j}")
                    continue
                image_inputs, input_ids, attention_mask, _ = batch
                image_inputs = image_inputs.to(device, non_blocking=True)
                input_ids    = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                # Forward pass (validation)
                logits = model(image_inputs=image_inputs, text_inputs={"input_ids": input_ids, "attention_mask": attention_mask})
                val_loss = compute_contrastive_loss(logits, temperature=1.0)
                val_losses.append(val_loss.item())
                all_metrics.append(compute_recall(logits))
        # Aggregate and log validation results (only on rank 0)
        if rank == 0:
            if len(val_losses) > 0:
                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
                wandb.log({"val_loss": avg_val_loss, **avg_metrics, "epoch": epoch})
            else:
                print("[rank0] Warning: No validation loss recorded (all val batches skipped).")
    
    if rank == 0:
        torch.save(model.module.state_dict(), f"clip_epoch_{epoch}.pth") 


    if rank == 0:
        wandb.finish()
    # Clean up distributed training environment
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
