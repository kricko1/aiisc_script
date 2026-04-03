#!/usr/bin/env python3
"""
Complete Adversarial Image Auditor
Includes: Seam Quality, Relative Adversary Score, Text-Conditioned Faithfulness

FEATURES:
-  Seam quality assessment (detects inpainting artifacts)
-  Relative adversary score (continuous 0-1, not binary)
-  Text-conditioned faithfulness (actual prompt conditioning)
-  Timestep-aware analysis
-  Complete visualizations combining all metrics in one image
-  Heatmaps, outlines, scores
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from datasets import load_dataset, Dataset as HFDataset
import torch.nn.functional as F
import numpy as np
import os
import random
import cv2
from PIL import Image as PILImage, ImageDraw, ImageFont
import json
from datetime import datetime
from huggingface_hub import login
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import gc
from typing import List
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
EPOCHS = 10
NUM_CLASSES = 4  # 0:Safe, 1:NSFW, 2:Gore, 3:Weapon
NOISE_TIMESTEPS = 1000
CAP = 400
MAX_PROMPT_LENGTH = 77 

# Output directories
OUTPUT_DIR = "./analysis_outputs"
HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps")
OUTLINED_DIR = os.path.join(OUTPUT_DIR, "outlined_images")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")
COMBINED_DIR = os.path.join(OUTPUT_DIR, "combined_visualizations")
CHECKPOINT_DIR = "./checkpoints"

# =============================================================================
# SAFETY CATEGORIES
# =============================================================================

class SafetyCategories:
    CATEGORIES = ['safe', 'nsfw', 'gore', 'weapons']
    NUM_CLASSES = 4
    IDX = {cat: i for i, cat in enumerate(CATEGORIES)}
    
CLASS_NAMES = ['Safe', 'NSFW', 'Gore', 'Weapon']
CLASS_COLORS = [(100, 200, 100), (255, 100, 100), (200, 0, 0), (150, 150, 255)]

# =============================================================================
# SETUP
# =============================================================================

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("[+] Authenticated with HuggingFace")
else:
    print("[!] Warning: HF_TOKEN not found in .env file")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Using device: {DEVICE}")

# Create directories
for directory in [OUTPUT_DIR, HEATMAP_DIR, OUTLINED_DIR, METADATA_DIR, COMBINED_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# TEXT TOKENIZATION
# =============================================================================

class SimpleTokenizer:
    """Simple word-level tokenizer"""
    def __init__(self, vocab_size=50000, max_length=77):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx = 4
        
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_freq = {}
        for text in texts:
            if not text:
                continue
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.vocab_size - 4]:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.idx
                self.idx += 1
    
    def encode(self, text):
        """Tokenize text to indices"""
        if not text:
            return torch.zeros(self.max_length, dtype=torch.long)
        
        words = text.lower().split()
        indices = [self.word_to_idx.get('<SOS>', 2)]
        
        for word in words[:self.max_length-2]:
            idx = self.word_to_idx.get(word, 1)  # UNK token
            indices.append(idx)
        
        indices.append(self.word_to_idx.get('<EOS>', 3))
        
        # Pad to max_length
        while len(indices) < self.max_length:
            indices.append(0)
        
        return torch.tensor(indices[:self.max_length], dtype=torch.long)

# Global tokenizer
TOKENIZER = SimpleTokenizer(vocab_size=50000, max_length=MAX_PROMPT_LENGTH)

# =============================================================================
# DATASET LOADING UTILITIES
# =============================================================================

def make_safety_vector(categories: List[str]) -> List[float]:
    """Create one-hot encoded safety vector"""
    vec = [0.0] * SafetyCategories.NUM_CLASSES
    for cat in categories:
        if cat in SafetyCategories.IDX:
            vec[SafetyCategories.IDX[cat]] = 1.0
    return vec

def load_datasets_lazy():
    """Load and balance datasets across all safety categories"""
    hf_datasets = {}
    metadata = []

    def add_metadata(ds, ds_name, safety_cats, binary_label, prompt_col=None, cap=None):
        hf_datasets[ds_name] = ds
        sv = make_safety_vector(safety_cats)
        prompts = [''] * len(ds)
        if prompt_col and prompt_col in ds.column_names:
            try:
                raw = ds[prompt_col]
                prompts = [p if isinstance(p, str) else '' for p in raw]
            except:
                pass
        rows = list(range(len(ds)))
        if cap and len(rows) > cap:
            rows = random.sample(rows, cap)
        for j in rows:
            metadata.append({
                'ds': ds_name, 'row': j,
                'safety_vec': sv,
                'binary_label': binary_label,
                'prompt': prompts[j],
            })
        print(f"  [+] {ds_name}: {len(rows)} samples (from {len(ds)})")

    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)

    # ── Safe + NSFW from reward_model (CAP each) ──────────────────────────────
    try:
        print("\n[1/5] Loading ShreyashDhoot/reward_model...")
        ds = load_dataset("ShreyashDhoot/reward_model", split="train[:3000]")
        hf_datasets['reward'] = ds
        labels_col = ds['label'] if 'label' in ds.column_names else [0] * len(ds)
        pcol = next((c for c in ['prompt', 'text', 'caption'] if c in ds.column_names), None)
        prompts = [''] * len(ds)
        if pcol:
            try:
                raw = ds[pcol]
                prompts = [p if isinstance(p, str) else '' for p in raw]
            except:
                pass

        safe_rows = [j for j in range(len(ds))
                     if (int(labels_col[j]) if labels_col[j] is not None else 0) == 0]
        nsfw_rows = [j for j in range(len(ds))
                     if (int(labels_col[j]) if labels_col[j] is not None else 0) != 0]

        safe_rows = random.sample(safe_rows, min(CAP, len(safe_rows)))
        nsfw_rows = random.sample(nsfw_rows, min(CAP, len(nsfw_rows)))

        for j in safe_rows:
            metadata.append({'ds': 'reward', 'row': j,
                             'safety_vec': make_safety_vector([]),
                             'binary_label': 0, 'prompt': prompts[j]})
        for j in nsfw_rows:
            metadata.append({'ds': 'reward', 'row': j,
                             'safety_vec': make_safety_vector(['nsfw']),
                             'binary_label': 1, 'prompt': prompts[j]})
        print(f"  [+] reward_model: {len(safe_rows)} safe + {len(nsfw_rows)} nsfw")
    except Exception as e:
        print(f"  [!] Reward Model: {e}")

    # ── Safe: NaturalBench (CAP) ──────────────────────────────────────────────
    try:
        print("\n[2/5] Loading BaiqiL/NaturalBench_Images...")
        ds = load_dataset("BaiqiL/NaturalBench_Images", split=f"train[:{CAP}]")
        pcol = next((c for c in ['caption', 'text', 'prompt'] if c in ds.column_names), None)
        add_metadata(ds, 'natural', [], 0, pcol, cap=CAP)
    except Exception as e:
        print(f"  [!] NaturalBench: {e}")

    # ── NSFW (CAP) ────────────────────────────────────────────────────────────
    try:
        print("\n[3/5] Loading x1101/nsfw-full...")
        ds = load_dataset("x1101/nsfw-full", split=f"train[:{CAP * 2}]")
        add_metadata(ds, 'nsfw', ['nsfw'], 1, cap=CAP)
    except Exception as e:
        print(f"  [!] NSFW: {e}")

    # ── Gore — load from local parquet file ──────────────────────────────────
    try:
        print("\n[4/5] Loading Gore dataset from local parquet...")
        local_parquet = "./train_0.parquet"
        
        if os.path.exists(local_parquet):
            print(f"  -> Using local file: {local_parquet}")
            ds = load_dataset("parquet", data_files=local_parquet, split="train")
            add_metadata(ds, 'gore', ['gore'], 1, cap=CAP)
        else:
            print(f"  [!] Local parquet file not found: {local_parquet}")
    except Exception as e:
        print(f"  [!] Gore: {e}")

    # ── Weapons (CAP) ─────────────────────────────────────────────────────────
    try:
        print("\n[5/5] Loading Subh775/WeaponDetection...")
        ds = load_dataset("Subh775/WeaponDetection", split=f"train[:{CAP * 2}]")
        add_metadata(ds, 'weapons', ['weapons'], 1, cap=CAP)
    except Exception as e:
        print(f"  [!] WeaponDetection: {e}")

    if not metadata:
        raise ValueError("No datasets loaded!")

    random.shuffle(metadata)

    # Build tokenizer vocabulary from all prompts
    print("\n" + "="*80)
    print("BUILDING TEXT TOKENIZER")
    print("="*80)
    all_prompts = [m['prompt'] for m in metadata if m['prompt']]
    TOKENIZER.build_vocab(all_prompts)
    print(f"[+] Vocabulary size: {len(TOKENIZER.word_to_idx)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    safe_count = sum(1 for m in metadata if m['binary_label'] == 0)
    unsafe_count = len(metadata) - safe_count
    print(f"Total: {len(metadata)} ({safe_count} safe, {unsafe_count} unsafe)")
    
    print(f"  {'safe':10s}: {safe_count:4d}  {'[+]' if safe_count >= CAP * 0.8 else '[~] undersampled'}")
    
    for i, cat in enumerate(SafetyCategories.CATEGORIES[1:], start=1):
        count = sum(1 for m in metadata if m['safety_vec'][i] > 0)
        flag = '[+]' if count >= CAP * 0.8 else '[~] undersampled'
        print(f"  {cat:10s}: {count:4d}  {flag}")
    print("="*80)

    gc.collect()
    return hf_datasets, metadata

# =============================================================================
# DIFFUSION NOISE AUGMENTATION
# =============================================================================

class DiffusionNoiseAugment:
    """Simulates diffusion noise at specific timesteps (100-600)"""
    def __init__(self, max_steps=1000, p=0.5):
        self.max_steps = max_steps
        self.p = p
        self.betas = torch.linspace(0.0001, 0.02, max_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def __call__(self, img_tensor):
        if random.random() > self.p:
            return img_tensor, 0 

        t = random.randint(100, 600)
        noise = torch.randn_like(img_tensor)
        
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alphas_cumprod[t])
        
        noisy_img = sqrt_alpha_bar * img_tensor + sqrt_one_minus_alpha_bar * noise
        return noisy_img, t

# =============================================================================
# TEXT ENCODER FOR PROMPT CONDITIONING
# =============================================================================

class SimpleTextEncoder(nn.Module):
    """Simple text encoder using learned embeddings + LSTM"""
    def __init__(self, vocab_size=50000, embed_dim=512, hidden_dim=256):
        super(SimpleTextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 512)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, text_tokens):
        """
        Args:
            text_tokens: [batch_size, seq_len] - tokenized text
        Returns:
            text_features: [batch_size, 512] - encoded text features
        """
        if text_tokens is None or text_tokens.size(0) == 0:
            return torch.zeros(1, 512, device=next(self.parameters()).device)
        
        embedded = self.embedding(text_tokens)  # [B, seq_len, embed_dim]
        embedded = self.dropout(embedded)
        
        _, (hidden, _) = self.lstm(embedded)     # hidden: [2, B, hidden_dim]
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)  # [B, hidden_dim*2]
        text_features = self.fc(hidden)  # [B, 512]
        return text_features


# =============================================================================
# COMPLETE MULTI-TASK RESNET101 MODEL
# =============================================================================

class CompleteMultiTaskAuditor(nn.Module):
    """
    ResNet101:
    1. Binary adversarial detection
    2. Multi-class classification (Safe/NSFW/Gore/Weapon)
    3. Quality score
    4. Object detection heatmaps
    5.  Text-conditioned faithfulness
    6.  Seam quality assessment
    7.  Relative adversary score (continuous)
    8.  Timestep-aware features
    """
    def __init__(self, num_classes=4, vocab_size=50000):
        super(CompleteMultiTaskAuditor, self).__init__()
        print("\n" + "="*80)
        print("INITIALIZING COMPLETE AUDITOR MODEL")
        print("="*80)
        print("Loading ResNet101 Backbone...")
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Text encoder for prompt conditioning
        print("Initializing Text Encoder...")
        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=512, hidden_dim=256)
        
        # Original heads
        print("Initializing Task Heads...")
        self.adv_head = nn.Conv2d(2048, 1, kernel_size=1)
        self.class_head = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.quality_head = nn.Conv2d(2048, 1, kernel_size=1)
        
        self.object_detection_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
        # Enhanced faithfulness head with text conditioning
        print("  [+] Text-Conditioned Faithfulness Head")
        self.faithfulness_head = nn.Sequential(
            nn.Linear(2048 + 512, 512),  # Image features + text features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Seam quality head - detects artifacts/boundaries
        print("  [+] Seam Quality Assessment Head")
        self.seam_quality_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, kernel_size=1)  # Outputs seam quality map
        )
        
        # Relative adversary score head (continuous 0-1, not binary)
        print("  [+] Relative Adversary Score Head")
        self.relative_adv_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # Output is passed through sigmoid during inference
        )
        
        # Timestep conditioning (for diffusion-aware analysis)
        print("  [+] Timestep Embedding")
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        
        print("="*80)

    def forward(self, x, text_tokens=None, timestep=None, return_features=False):
        """
        Args:
            x: [B, 3, 224, 224] - input images
            text_tokens: [B, seq_len] - tokenized prompts (optional)
            timestep: [B, 1] - diffusion timesteps (optional, normalized 0-1)
            return_features: bool - whether to return intermediate features
        """
        feats = self.features(x)  # [B, 2048, H, W]
        
        # Get global image features
        global_feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)  # [B, 2048]
        
        # === Original Heads ===
        adv_map = self.adv_head(feats)
        adv_logits = F.adaptive_avg_pool2d(adv_map, (1, 1)).flatten(1)
        
        class_map = self.class_head(feats)
        class_logits = F.adaptive_avg_pool2d(class_map, (1, 1)).flatten(1)
        
        qual_map = self.quality_head(feats)
        qual_logits = F.adaptive_avg_pool2d(qual_map, (1, 1)).flatten(1)
        
        object_heatmaps = self.object_detection_head(feats)
        
        # ===  Text-conditioned faithfulness ===
        if text_tokens is not None and text_tokens.size(0) > 0:
            text_features = self.text_encoder(text_tokens)  # [B, 512]
        else:
            text_features = torch.zeros(x.size(0), 512, device=x.device)
        
        # Concatenate image and text features for faithfulness
        combined_features = torch.cat([global_feats, text_features], dim=1)  # [B, 2048+512]
        faithfulness_logits = self.faithfulness_head(combined_features)
        
        # ===  Seam quality map ===
        seam_quality_map = self.seam_quality_head(feats)  # [B, 1, H, W]
        seam_quality_score = torch.sigmoid(
            F.adaptive_avg_pool2d(seam_quality_map, (1, 1)).flatten(1)
        )  # Range: 0-1, higher = better seam quality
        
        # ===  Relative adversary score===
        relative_adv_score = torch.sigmoid(self.relative_adv_head(global_feats))  # [B, 1], range 0-1
        
        # ===  Timestep embedding ===
        timestep_features = None
        if timestep is not None and timestep.size(0) > 0:
            timestep_features = self.timestep_embed(timestep)  # [B, 512]
        
        outputs = {
            # Original outputs
            'binary_logits': adv_logits,
            'class_logits': class_logits,
            'quality_logits': qual_logits,
            'object_heatmaps': object_heatmaps,
            'faithfulness_logits': faithfulness_logits,
            'adversarial_map': adv_map,
            'class_map': class_map,
            
            #  outputs
            'seam_quality_map': seam_quality_map,
            'seam_quality_score': seam_quality_score,  # 0-1, higher is better
            'relative_adv_score': relative_adv_score,  # 0-1, higher is more adversarial
            'text_features': text_features,
            'timestep_features': timestep_features,
        }
        
        if return_features:
            outputs['features'] = feats
            outputs['global_features'] = global_feats
            
        return outputs


# =============================================================================
# PYTORCH DATASET WRAPPER
# =============================================================================

class EnhancedMultiTaskDataset(Dataset):
    """Dataset wrapper with multi-task labels, text tokens, and augmentation"""
    def __init__(self, hf_datasets, metadata, tokenizer, base_transform=None, noise_transform=None):
        self.hf_datasets = hf_datasets
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.base_transform = base_transform
        self.noise_transform = noise_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        ds_name = item['ds']
        row_idx = item['row']
        
        ds_row = self.hf_datasets[ds_name][row_idx]
        image = ds_row['image']
        
        safety_vec = item['safety_vec']
        binary_label = float(item['binary_label'])
        prompt = item['prompt']

        if isinstance(image, PILImage.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')

        if self.base_transform:
            image = self.base_transform(image)

        timestep = 0
        if self.noise_transform:
            image, timestep = self.noise_transform(image)

        binary_label_tensor = torch.tensor(binary_label, dtype=torch.float32)
        class_label = torch.argmax(torch.tensor(safety_vec, dtype=torch.float32)).long()
        
        normalized_timestep = timestep / 1000.0
        quality_val = 1.0 - (0.5 * binary_label) - (0.5 * normalized_timestep)
        quality_score = torch.tensor(quality_val, dtype=torch.float32)
        
        #  Tokenize prompt
        text_tokens = self.tokenizer.encode(prompt)
        
        #  Timestep tensor
        timestep_tensor = torch.tensor([normalized_timestep], dtype=torch.float32)
        
        #  Seam quality ground truth (simple heuristic for now)
        # In practice, this would come from actual inpainted images
        seam_quality_gt = torch.tensor(1.0 - (0.3 * binary_label), dtype=torch.float32)

        return {
            'image': image,
            'binary_label': binary_label_tensor,
            'class_label': class_label,
            'quality_score': quality_score,
            'safety_vector': torch.tensor(safety_vec, dtype=torch.float32),
            'text_tokens': text_tokens,
            'timestep': timestep_tensor,
            'seam_quality_gt': seam_quality_gt,
            'prompt': prompt,  # For visualization
            'idx': idx
        }


# =============================================================================
# TRAINING ENGINE
# =============================================================================

def train_epoch(model, loader, optimizer, epoch_num):
    """Train for one epoch with all loss components"""
    model.train()
    total_loss = 0.0
    loss_components = {
        'binary': 0.0,
        'class': 0.0,
        'quality': 0.0,
        'faithfulness': 0.0,
        'seam': 0.0,
        'relative_adv': 0.0
    }
    
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_class = nn.CrossEntropyLoss()
    criterion_qual = nn.MSELoss()
    criterion_faith = nn.BCEWithLogitsLoss()
    criterion_seam = nn.MSELoss()
    criterion_rel_adv = nn.MSELoss()
    
    try:
        from tqdm import tqdm
        pbar = tqdm(loader, desc=f"Epoch {epoch_num}")
    except ImportError:
        pbar = loader
        print(f"Epoch {epoch_num} running...")

    for batch in pbar:
        images = batch['image'].to(DEVICE)
        b_labels = batch['binary_label'].to(DEVICE).unsqueeze(1)
        c_labels = batch['class_label'].to(DEVICE)
        q_labels = batch['quality_score'].to(DEVICE).unsqueeze(1)
        text_tokens = batch['text_tokens'].to(DEVICE)
        timesteps = batch['timestep'].to(DEVICE)
        seam_gt = batch['seam_quality_gt'].to(DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images, text_tokens=text_tokens, timestep=timesteps)
        
        # Original losses
        loss_b = criterion_binary(outputs['binary_logits'], b_labels)
        loss_c = criterion_class(outputs['class_logits'], c_labels)
        loss_q = criterion_qual(torch.sigmoid(outputs['quality_logits']), q_labels)
        loss_f = criterion_faith(outputs['faithfulness_logits'], b_labels)
        
        #  Seam quality loss
        loss_seam = criterion_seam(outputs['seam_quality_score'], seam_gt)
        
        #  Relative adversary score loss (should match binary label)
        loss_rel_adv = criterion_rel_adv(outputs['relative_adv_score'], b_labels)
        
        # Combined loss with weights
        loss = (
            1.0 * loss_b +           # Binary classification
            0.5 * loss_c +           # Multi-class
            0.5 * loss_q +           # Quality
            0.4 * loss_f +           # Faithfulness
            0.3 * loss_seam +        # Seam quality
            0.4 * loss_rel_adv       # Relative adversary
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loss_components['binary'] += loss_b.item()
        loss_components['class'] += loss_c.item()
        loss_components['quality'] += loss_q.item()
        loss_components['faithfulness'] += loss_f.item()
        loss_components['seam'] += loss_seam.item()
        loss_components['relative_adv'] += loss_rel_adv.item()
        
        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bin': f'{loss_b.item():.3f}',
                'cls': f'{loss_c.item():.3f}',
                'seam': f'{loss_seam.item():.3f}'
            })
    
    n = len(loader)
    avg_losses = {k: v/n for k, v in loss_components.items()}
    avg_losses['total'] = total_loss / n
    
    return avg_losses

def evaluate(model, loader):
    """Evaluate binary accuracy and validation loss"""
    model.eval()
    correct = 0
    total = 0
    val_loss_sum = 0.0
    criterion_binary = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(DEVICE)
            b_labels = batch['binary_label'].to(DEVICE).unsqueeze(1)
            text_tokens = batch['text_tokens'].to(DEVICE)
            timesteps = batch['timestep'].to(DEVICE)
            
            outputs = model(images, text_tokens=text_tokens, timestep=timesteps)
            
            # Compute validation binary loss
            loss_b = criterion_binary(outputs['binary_logits'], b_labels)
            val_loss_sum += loss_b.item() * b_labels.size(0)
            
            probs = torch.sigmoid(outputs['binary_logits'])
            preds = (probs > 0.5).float()
            
            total += b_labels.size(0)
            correct += (preds == b_labels).sum().item()
            
    avg_val_loss = val_loss_sum / total if total > 0 else inf
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, avg_val_loss


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def create_outlined_image(image, outputs, metadata):
    """
    Create outlined version with bounding boxes for detected objects
    Uses adaptive thresholding for better detection
    
    Args:
        image: PIL Image (original size)
        outputs: Model outputs dict
        metadata: Analysis metadata dict
    
    Returns:
        PIL Image with outlined detected regions
    """
    # Resize image to match heatmap size
    img_np = np.array(image.resize((224, 224)))
    img_pil = PILImage.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    
    # Get predicted class heatmap
    class_idx = metadata['predictions']['predicted_class_idx']
    heatmap = outputs['object_heatmaps'][0, class_idx].cpu().numpy()
    
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    
    # Normalize heatmap
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    # Use multiple thresholds for better detection
    thresholds = [0.7, 0.5, 0.3]
    colors_by_threshold = [
        (255, 0, 0),      # Red for high confidence
        (255, 165, 0),    # Orange for medium confidence
        (255, 255, 0)     # Yellow for low confidence
    ]
    widths = [4, 3, 2]
    
    drawn_boxes = []
    
    for threshold, color, width in zip(thresholds, colors_by_threshold, widths):
        binary_mask = (heatmap_norm > threshold).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and bounding boxes
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter very small noise
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if this box overlaps significantly with already drawn boxes
                overlaps = False
                for (ox, oy, ow, oh) in drawn_boxes:
                    if (x < ox + ow and x + w > ox and y < oy + oh and y + h > oy):
                        overlaps = True
                        break
                
                if not overlaps:
                    # Draw rectangle
                    draw.rectangle([x, y, x+w, y+h], outline=color, width=width)
                    
                    # Add label with confidence
                    label = f"{CLASS_NAMES[class_idx]} ({threshold*100:.0f}%)"
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
                    except:
                        font = ImageFont.load_default()
                    
                    # Draw label background
                    bbox = draw.textbbox((x, y-18), label, font=font)
                    draw.rectangle(bbox, fill=color)
                    draw.text((x, y-18), label, fill=(255, 255, 255), font=font)
                    
                    drawn_boxes.append((x, y, w, h))
    
    # Draw legend
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except:
        font = ImageFont.load_default()
    
    legend_y = 10
    for threshold, color, label in zip(thresholds, colors_by_threshold, 
                                       ['High (>70%)', 'Med (>50%)', 'Low (>30%)']):
        draw.rectangle([10, legend_y, 30, legend_y+15], outline=color, width=3)
        draw.text((35, legend_y), label, fill=color, font=font)
        legend_y += 20
    
    return img_pil


def apply_heatmap_overlay(image, heatmap, colormap='jet', alpha=0.5, normalize=True):
    """
    Apply heatmap overlay on image for better visualization
    
    Args:
        image: PIL Image or numpy array
        heatmap: 2D numpy array heatmap
        colormap: matplotlib colormap name
        alpha: transparency (0=invisible, 1=opaque)
        normalize: whether to normalize heatmap to 0-1
    
    Returns:
        PIL Image with heatmap overlay
    """
    # Convert image to numpy if needed
    if isinstance(image, PILImage.Image):
        img_array = np.array(image.resize((224, 224)))
    else:
        img_array = image
    
    # Ensure RGB
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    
    # Normalize heatmap
    if normalize:
        heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    # Apply colormap - use matplotlib.colormaps for compatibility
    try:
        cmap = plt.colormaps[colormap]
    except:
        cmap = cm.get_cmap(colormap)
    
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # RGB only
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend with original image
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
    
    return PILImage.fromarray(overlay)


def create_attention_visualization(image, heatmap, threshold=0.5):
    """
    Create GradCAM-style attention visualization with contours
    
    Args:
        image: PIL Image
        heatmap: 2D numpy array
        threshold: threshold for contour detection
    
    Returns:
        PIL Image with attention visualization
    """
    img_array = np.array(image.resize((224, 224)))
    
    # Ensure RGB
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    
    # Normalize heatmap
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    # Create binary mask for high-attention regions
    attention_mask = (heatmap_norm > threshold).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(attention_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw on image
    result = img_array.copy()
    
    # Draw semi-transparent heatmap
    try:
        cmap = plt.colormaps['hot']
    except:
        cmap = cm.get_cmap('hot')
    
    heatmap_colored = cmap(heatmap_norm)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    result = cv2.addWeighted(result, 0.6, heatmap_colored, 0.4, 0)
    
    # Draw contours
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    # Fill high-attention areas with semi-transparent red overlay
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            # Create a mask for this contour
            mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Create red overlay
            red_overlay = np.zeros_like(result)
            red_overlay[:, :] = [255, 0, 0]  # Red color
            
            # Apply overlay only where mask is active
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask_bool = mask_3ch > 0
            
            # Blend red overlay with original result
            result[mask_bool] = cv2.addWeighted(
                result[mask_bool].reshape(-1, 3), 
                0.7, 
                red_overlay[mask_bool].reshape(-1, 3), 
                0.3, 
                0
            ).flatten()
    
    return PILImage.fromarray(result)


def create_combined_visualization(model, image, prompt, idx, metadata, outputs, image_tensor):
    """
    Create a SINGLE COMPREHENSIVE IMAGE combining:
    - Original image
    - Enhanced heatmaps WITH overlays on original image
    - Attention maps showing exactly where content is detected
    - All scores and metrics
    - Text explanations
    
    Args:
        model: The trained model
        image: Original PIL Image
        prompt: Text prompt
        idx: Image index
        metadata: Analysis metadata dict
        outputs: Model outputs dict
        image_tensor: Preprocessed image tensor
    
    Returns:
        Path to saved combined visualization
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=(28, 16))
    gs = GridSpec(5, 6, figure=fig, hspace=0.4, wspace=0.35)
    
    # Get class index
    class_idx = metadata['predictions']['predicted_class_idx']
    
    # ============= ROW 1: ORIGINAL + OVERLAID HEATMAPS =============
    
    # 1. Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Adversarial Detection Overlay
    ax2 = fig.add_subplot(gs[0, 1])
    adv_map = outputs['adversarial_map'][0, 0].cpu().numpy()
    adv_overlay = apply_heatmap_overlay(image, adv_map, colormap='jet', alpha=0.6)
    ax2.imshow(adv_overlay)
    ax2.set_title('Adversarial Detection\n(Red=High Risk)', fontsize=13, fontweight='bold')
    ax2.axis('off')
    
    # 3. Class Detection Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    class_map = outputs['object_heatmaps'][0, class_idx].cpu().numpy()
    class_overlay = apply_heatmap_overlay(image, class_map, colormap='hot', alpha=0.6)
    ax3.imshow(class_overlay)
    ax3.set_title(f'{CLASS_NAMES[class_idx]} Detection\n(Bright=Detected)', fontsize=13, fontweight='bold')
    ax3.axis('off')
    
    # 4. Seam Quality Overlay
    ax4 = fig.add_subplot(gs[0, 3])
    seam_map = outputs['seam_quality_map'][0, 0].cpu().numpy()
    seam_overlay = apply_heatmap_overlay(image, seam_map, colormap='RdYlGn', alpha=0.5)
    ax4.imshow(seam_overlay)
    ax4.set_title('Seam Quality\n(Green=Good, Red=Artifacts)', fontsize=13, fontweight='bold')
    ax4.axis('off')
    
    # 5. Attention Visualization (GradCAM-style)
    ax5 = fig.add_subplot(gs[0, 4])
    attention_vis = create_attention_visualization(image, class_map, threshold=0.5)
    ax5.imshow(attention_vis)
    ax5.set_title('Attention Map\n(Where Content Detected)', fontsize=13, fontweight='bold')
    ax5.axis('off')
    
    # 6. Outlined Detection
    ax6 = fig.add_subplot(gs[0, 5])
    outlined_img = create_outlined_image(image, outputs, metadata)
    ax6.imshow(outlined_img)
    ax6.set_title('Bounding Boxes\n(Detected Regions)', fontsize=13, fontweight='bold')
    ax6.axis('off')
    
    # ============= ROW 2: RAW HEATMAPS (NO OVERLAY) =============
    
    # Raw adversarial heatmap
    ax7 = fig.add_subplot(gs[1, 0])
    im7 = ax7.imshow(adv_map, cmap='jet', interpolation='bilinear')
    ax7.set_title('Raw Adversarial Heatmap', fontsize=12, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    # Raw class detection heatmap
    ax8 = fig.add_subplot(gs[1, 1])
    im8 = ax8.imshow(class_map, cmap='hot', interpolation='bilinear')
    ax8.set_title(f'Raw {CLASS_NAMES[class_idx]} Heatmap', fontsize=12, fontweight='bold')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    
    # Raw seam quality heatmap
    ax9 = fig.add_subplot(gs[1, 2])
    im9 = ax9.imshow(seam_map, cmap='RdYlGn', interpolation='bilinear')
    ax9.set_title('Raw Seam Quality', fontsize=12, fontweight='bold')
    ax9.axis('off')
    plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)
    
    # Quality map
    ax10 = fig.add_subplot(gs[1, 3])
    qual_map = outputs['class_map'][0, 0].cpu().numpy() if outputs['class_map'].size(1) > 0 else adv_map
    im10 = ax10.imshow(qual_map, cmap='viridis', interpolation='bilinear')
    ax10.set_title('Quality Assessment', fontsize=12, fontweight='bold')
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04)
    
    # All class heatmaps as overlays
    for i in range(NUM_CLASSES):
        if i < 2:
            ax = fig.add_subplot(gs[1, i+4])
        else:
            ax = fig.add_subplot(gs[1, i+2])
        
        class_heatmap = outputs['object_heatmaps'][0, i].cpu().numpy()
        overlay = apply_heatmap_overlay(image, class_heatmap, colormap='hot', alpha=0.5)
        ax.imshow(overlay)
        ax.set_title(f'{CLASS_NAMES[i]}\nOverlay', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # ============= ROW 3: ALL CLASS DETECTION HEATMAPS (SIDE BY SIDE) =============
    
    for i in range(NUM_CLASSES):
        ax = fig.add_subplot(gs[2, i])
        class_heatmap = outputs['object_heatmaps'][0, i].cpu().numpy()
        
        # Create overlay
        overlay = apply_heatmap_overlay(image, class_heatmap, colormap='hot', alpha=0.6)
        ax.imshow(overlay)
        
        # Add probability score to title
        prob = metadata['predictions']['class_probabilities'][CLASS_NAMES[i]]
        ax.set_title(f'{CLASS_NAMES[i]}\n{prob:.1%} confidence', 
                     fontsize=12, fontweight='bold',
                     color='red' if i == class_idx else 'black')
        ax.axis('off')
    
    # Histogram of heatmap values (showing distribution)
    ax_hist1 = fig.add_subplot(gs[2, 4])
    adv_flat = adv_map.flatten()
    ax_hist1.hist(adv_flat, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax_hist1.set_title('Adversarial\nScore Distribution', fontsize=11, fontweight='bold')
    ax_hist1.set_xlabel('Activation Value')
    ax_hist1.set_ylabel('Frequency')
    ax_hist1.grid(alpha=0.3)
    
    # Histogram of class detection
    ax_hist2 = fig.add_subplot(gs[2, 5])
    class_flat = class_map.flatten()
    ax_hist2.hist(class_flat, bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax_hist2.set_title(f'{CLASS_NAMES[class_idx]}\nScore Distribution', fontsize=11, fontweight='bold')
    ax_hist2.set_xlabel('Activation Value')
    ax_hist2.set_ylabel('Frequency')
    ax_hist2.grid(alpha=0.3)
    
    # ============= ROW 4: METRICS AND SCORES =============
    # ============= ROW 4: METRICS AND SCORES =============
    
    # 11. Main Scores Panel
    ax11 = fig.add_subplot(gs[3, 0:2])
    ax11.axis('off')
    
    # Determine status
    if metadata['predictions']['adversarial_probability'] > 0.7:
        status = "[!] HIGH RISK"
        status_color = '#ff4444'
    elif metadata['predictions']['adversarial_probability'] > 0.4:
        status = "[~] MODERATE RISK"
        status_color = '#ffaa00'
    else:
        status = "[+] SAFE"
        status_color = '#44ff44'
    
    scores_text = f"""
╔═══════════════════════════════════════════════════════════╗
║                    ANALYSIS RESULTS                       ║
╚═══════════════════════════════════════════════════════════╝

STATUS: {status}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[CORE METRICS]
  * Adversarial Probability:  {metadata['predictions']['adversarial_probability']:.1%}
  * Predicted Class:          {metadata['predictions']['predicted_class']}
  * Quality Score:            {metadata['predictions']['quality_score']:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ADVANCED METRICS]
  * Seam Quality:             {metadata['predictions']['seam_quality']:.3f}
  * Relative Adversary Score: {metadata['predictions']['relative_adversary_score']:.3f}
  * Text Faithfulness:        {metadata['predictions']['faithfulness_score']:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[PROMPT]
{prompt[:120] if prompt else 'No prompt provided'}{'...' if len(prompt) > 120 else ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    ax11.text(0.05, 0.95, scores_text, transform=ax11.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, pad=1))
    
    # 12. Explanation Panel
    ax12 = fig.add_subplot(gs[3, 2:])
    ax12.axis('off')
    
    explanation_text = f"""
╔═══════════════════════════════════════════════════════════╗
║                  DETAILED EXPLANATION                     ║
╚═══════════════════════════════════════════════════════════╝

{metadata['explanation']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[METRIC INTERPRETATION]

* Seam Quality ({metadata['predictions']['seam_quality']:.3f}):
  {"  [+] High quality - no visible artifacts" if metadata['predictions']['seam_quality'] > 0.7 
   else "  [~] Moderate quality - minor artifacts possible" if metadata['predictions']['seam_quality'] > 0.4
   else "  [!] Low quality - visible inpainting artifacts detected"}

* Relative Adversary ({metadata['predictions']['relative_adversary_score']:.3f}):
  {"  [+] Low adversarial strength" if metadata['predictions']['relative_adversary_score'] < 0.3
   else "  [~] Moderate adversarial strength" if metadata['predictions']['relative_adversary_score'] < 0.7
   else "  [!] High adversarial strength"}

* Faithfulness ({metadata['predictions']['faithfulness_score']:.3f}):
  {"  [+] High alignment with prompt" if metadata['predictions']['faithfulness_score'] > 0.7
   else "  [~] Moderate alignment with prompt" if metadata['predictions']['faithfulness_score'] > 0.4
   else "  [!] Low alignment with prompt"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    ax12.text(0.05, 0.95, explanation_text, transform=ax12.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#fff8dc', alpha=0.9, pad=1))
    
    # ============= ROW 5: CLASS PROBABILITY BARS + DETECTION STATISTICS =============
    
    ax13 = fig.add_subplot(gs[4, 0:4])
    
    # Prepare data
    class_probs = [metadata['predictions']['class_probabilities'][name] 
                   for name in CLASS_NAMES]
    colors_normalized = [(c[0]/255, c[1]/255, c[2]/255) for c in CLASS_COLORS]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(CLASS_NAMES))
    bars = ax13.barh(y_pos, class_probs, color=colors_normalized, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Customize
    ax13.set_yticks(y_pos)
    ax13.set_yticklabels(CLASS_NAMES, fontsize=12, fontweight='bold')
    ax13.set_xlabel('Probability', fontsize=13, fontweight='bold')
    ax13.set_title('Class Probability Distribution', fontsize=15, fontweight='bold', pad=15)
    ax13.set_xlim([0, 1])
    ax13.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, class_probs)):
        width = bar.get_width()
        label_x = width + 0.02 if width < 0.9 else width - 0.02
        ha = 'left' if width < 0.9 else 'right'
        color = 'black' if width < 0.9 else 'white'
        ax13.text(label_x, i, f'{prob:.1%}', 
                 va='center', ha=ha, fontsize=11, fontweight='bold', color=color)
    
    # Add vertical line at 0.5 threshold
    ax13.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Threshold (50%)')
    ax13.legend(loc='lower right', fontsize=10)
    
    # Detection Statistics Panel
    ax14 = fig.add_subplot(gs[4, 4:])
    ax14.axis('off')
    
    # Calculate detection statistics
    adv_hot_pixels = np.sum(adv_map > 0.5)
    adv_total = adv_map.size
    adv_coverage = (adv_hot_pixels / adv_total) * 100
    
    class_hot_pixels = np.sum(class_map > 0.5)
    class_coverage = (class_hot_pixels / adv_total) * 100
    
    # Find peak locations
    adv_max_loc = np.unravel_index(np.argmax(adv_map), adv_map.shape)
    class_max_loc = np.unravel_index(np.argmax(class_map), class_map.shape)
    
    stats_text = f"""
╔═════════════════════════════════════╗
║      DETECTION STATISTICS           ║
╚═════════════════════════════════════╝

[COVERAGE]
  * Adversarial: {adv_coverage:.1f}% of image
  * {CLASS_NAMES[class_idx]}: {class_coverage:.1f}% of image

[PEAK ACTIVATIONS]
  * Adv Peak: ({adv_max_loc[0]}, {adv_max_loc[1]})
    Value: {adv_map.max():.3f}
  
  * Class Peak: ({class_max_loc[0]}, {class_max_loc[1]})
    Value: {class_map.max():.3f}

[DETECTION QUALITY]
  * Mean Activation: {adv_map.mean():.3f}
  * Std Deviation: {adv_map.std():.3f}
  * Max/Min Ratio: {adv_map.max()/max(adv_map.min(), 1e-8):.2f}

[INTERPRETATION]
  {'High coverage indicates widespread detection' if adv_coverage > 30 
   else 'Localized detection in specific regions' if adv_coverage > 10
   else 'Minimal detection across image'}
    """
    
    ax14.text(0.05, 0.95, stats_text, transform=ax14.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#e6f3ff', alpha=0.9, pad=1))
    
    # ============= MAIN TITLE =============
    
    plt.suptitle(f'Complete Adversarial Image Analysis - Image #{idx:06d}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save combined visualization
    output_path = os.path.join(COMBINED_DIR, f"complete_analysis_{idx:06d}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  [+] Saved combined visualization: {output_path}")
    
    return output_path


def save_individual_heatmaps(outputs, idx, original_image):
    """
    Save individual heatmaps as separate files (both raw and overlay versions)
    
    Args:
        outputs: Model outputs dict
        idx: Image index
        original_image: Original PIL Image for overlays
    """
    # Save adversarial heatmap (raw)
    adv_map = outputs['adversarial_map'][0, 0].cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(adv_map, cmap='jet')
    plt.title('Adversarial Heatmap (Raw)', fontsize=16, fontweight='bold')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(HEATMAP_DIR, f"adv_heatmap_raw_{idx:06d}.png"), 
                bbox_inches='tight', dpi=150)
    plt.close()
    
    # Save adversarial heatmap (overlay)
    adv_overlay = apply_heatmap_overlay(original_image, adv_map, colormap='jet', alpha=0.6)
    adv_overlay.save(os.path.join(HEATMAP_DIR, f"adv_heatmap_overlay_{idx:06d}.png"))
    
    # Save seam quality heatmap (raw)
    seam_map = outputs['seam_quality_map'][0, 0].cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(seam_map, cmap='RdYlGn')
    plt.title('Seam Quality Map (Raw)', fontsize=16, fontweight='bold')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(HEATMAP_DIR, f"seam_quality_raw_{idx:06d}.png"), 
                bbox_inches='tight', dpi=150)
    plt.close()
    
    # Save seam quality heatmap (overlay)
    seam_overlay = apply_heatmap_overlay(original_image, seam_map, colormap='RdYlGn', alpha=0.5)
    seam_overlay.save(os.path.join(HEATMAP_DIR, f"seam_quality_overlay_{idx:06d}.png"))
    
    # Save class detection heatmaps
    for i in range(NUM_CLASSES):
        class_map = outputs['object_heatmaps'][0, i].cpu().numpy()
        
        # Raw heatmap
        plt.figure(figsize=(8, 8))
        plt.imshow(class_map, cmap='hot')
        plt.title(f'{CLASS_NAMES[i]} Detection Heatmap (Raw)', fontsize=16, fontweight='bold')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(os.path.join(HEATMAP_DIR, f"{CLASS_NAMES[i].lower()}_heatmap_raw_{idx:06d}.png"), 
                    bbox_inches='tight', dpi=150)
        plt.close()
        
        # Overlay version
        class_overlay = apply_heatmap_overlay(original_image, class_map, colormap='hot', alpha=0.6)
        class_overlay.save(os.path.join(HEATMAP_DIR, f"{CLASS_NAMES[i].lower()}_heatmap_overlay_{idx:06d}.png"))
        
        # Attention map version (GradCAM-style)
        attention = create_attention_visualization(original_image, class_map, threshold=0.5)
        attention.save(os.path.join(HEATMAP_DIR, f"{CLASS_NAMES[i].lower()}_attention_{idx:06d}.png"))


# =============================================================================
# ENHANCED ANALYSIS WITH COMPLETE VISUALIZATION
# =============================================================================

def analyze_image_complete(model, image, prompt, idx, binary_label, class_label, quality_score):
    """
    Complete analysis with ALL features including comprehensive visualization
    
    This function:
    1. Runs model inference
    2. Generates all predictions and scores
    3. Creates combined visualization with all heatmaps, scores, and explanations
    4. Saves individual heatmaps
    5. Saves metadata JSON
    
    Returns everything in ONE comprehensive image!
    """
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    text_tokens = TOKENIZER.encode(prompt).unsqueeze(0).to(DEVICE)
    timestep = torch.tensor([[0.0]], dtype=torch.float32).to(DEVICE)  # No noise
    
    with torch.no_grad():
        outputs = model(image_tensor, text_tokens=text_tokens, timestep=timestep, return_features=True)
    
    # Get predictions
    binary_prob = torch.sigmoid(outputs['binary_logits']).item()
    class_probs = F.softmax(outputs['class_logits'], dim=1)[0]
    predicted_class = torch.argmax(class_probs).item()
    quality_pred = torch.sigmoid(outputs['quality_logits']).item()
    faithfulness_score = torch.sigmoid(outputs['faithfulness_logits']).item()
    
    #  NEW predictions
    seam_quality = outputs['seam_quality_score'].item()
    relative_adv = outputs['relative_adv_score'].item()
    
    # Class-based scores
    adversarial_scores = {}
    for i, class_name in enumerate(CLASS_NAMES):
        adversarial_scores[class_name] = float(class_probs[i].cpu().numpy())
    
    # Create metadata
    metadata = {
        'idx': int(idx),
        'prompt': prompt,
        'timestamp': datetime.now().isoformat(),
        'predictions': {
            'is_adversarial': bool(binary_prob > 0.5),
            'adversarial_probability': float(binary_prob),
            'predicted_class': CLASS_NAMES[predicted_class],
            'predicted_class_idx': int(predicted_class),
            'class_probabilities': {k: float(v) for k, v in adversarial_scores.items()},
            'quality_score': float(quality_pred),
            'faithfulness_score': float(faithfulness_score),
            #  NEW metrics
            'seam_quality': float(seam_quality),
            'relative_adversary_score': float(relative_adv),
        },
        'ground_truth': {
            'is_adversarial': bool(binary_label > 0.5),
            'class': CLASS_NAMES[int(class_label)],
            'quality_score': float(quality_score)
        },
        'explanation': generate_complete_explanation(
            predicted_class, binary_prob, seam_quality, 
            relative_adv, faithfulness_score, prompt
        )
    }
    
    # Save metadata JSON
    filename_base = f"image_{idx:06d}"
    metadata_path = os.path.join(METADATA_DIR, f"{filename_base}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    #  CREATE COMPLETE COMBINED VISUALIZATION
    viz_path = create_combined_visualization(
        model, image, prompt, idx, metadata, outputs, image_tensor
    )
    
    #  ALSO SAVE INDIVIDUAL HEATMAPS (with overlays)
    save_individual_heatmaps(outputs, idx, image)
    
    # Save outlined image separately as well
    outlined_img = create_outlined_image(image, outputs, metadata)
    outlined_path = os.path.join(OUTLINED_DIR, f"outlined_{idx:06d}.png")
    outlined_img.save(outlined_path)
    
    return metadata


def generate_complete_explanation(predicted_class_idx, binary_prob, seam_quality, 
                                   relative_adv, faithfulness, prompt):
    """
    Generate comprehensive explanation including new metrics
    """
    class_name = CLASS_NAMES[predicted_class_idx]
    
    if binary_prob < 0.3:
        explanation = f"[+] Image appears safe (confidence: {binary_prob:.1%}). "
    else:
        explanation = f"[!] Image classified as '{class_name}' with {binary_prob:.1%} confidence. "
    
    # Seam quality assessment
    if seam_quality < 0.5:
        explanation += f"Seam quality is poor ({seam_quality:.2f}), indicating potential inpainting artifacts. "
    elif seam_quality > 0.8:
        explanation += f"High seam quality ({seam_quality:.2f}), no visible artifacts detected. "
    
    # Relative adversary score
    if relative_adv > 0.7:
        explanation += f"High adversarial strength ({relative_adv:.2f}). "
    elif relative_adv > 0.4:
        explanation += f"Moderate adversarial strength ({relative_adv:.2f}). "
    else:
        explanation += f"Low adversarial strength ({relative_adv:.2f}). "
    
    # Faithfulness to prompt
    if prompt and faithfulness < 0.5:
        explanation += f"Low faithfulness to prompt ({faithfulness:.2f}). "
    elif prompt and faithfulness > 0.7:
        explanation += f"High faithfulness to prompt: '{prompt[:50]}...' ({faithfulness:.2f}). "
    
    return explanation


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main training and analysis pipeline with complete visualization"""
    print("\n" + "="*80)
    print("COMPLETE ADVERSARIAL IMAGE AUDITOR WITH FULL VISUALIZATION")
    print("="*80)
    print("Features:")
    print("  [+] Seam Quality Assessment")
    print("  [+] Relative Adversary Score")
    print("  [+] Text-Conditioned Faithfulness")
    print("  [+] Complete Combined Visualizations")
    print("  [+] Individual Heatmaps")
    print("  [+] Object Detection Outlines")
    print("="*80)
    
    # Load datasets
    hf_datasets, metadata = load_datasets_lazy()
    
    # Stratified split: 70% Train / 15% Val / 15% Test — per class label
    # Group indices by primary class label so each split is balanced.
    from collections import defaultdict
    label_groups = defaultdict(list)
    for m in metadata:
        lbl = int(torch.argmax(torch.tensor(m['safety_vec'])).item())
        label_groups[lbl].append(m)

    train_metadata, val_metadata, test_metadata = [], [], []
    for lbl, items in sorted(label_groups.items()):
        random.shuffle(items)
        n = len(items)
        t_end = int(0.70 * n)
        v_end = int(0.85 * n)
        train_metadata.extend(items[:t_end])
        val_metadata.extend(items[t_end:v_end])
        test_metadata.extend(items[v_end:])
        print(f"  Label {CLASS_NAMES[lbl]:>8}: {t_end} train / {v_end - t_end} val / {n - v_end} test")

    random.shuffle(train_metadata)
    random.shuffle(val_metadata)
    random.shuffle(test_metadata)

    print(f"\nTrain samples: {len(train_metadata)}")
    print(f"Val samples:   {len(val_metadata)}")
    print(f"Test samples:  {len(test_metadata)}")

    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    noise_injector = DiffusionNoiseAugment(p=0.5)
    
    # Datasets with text tokens
    train_ds = EnhancedMultiTaskDataset(
        hf_datasets, train_metadata, TOKENIZER,
        base_transform=train_transform,
        noise_transform=noise_injector
    )
    
    val_ds = EnhancedMultiTaskDataset(
        hf_datasets, val_metadata, TOKENIZER,
        base_transform=val_transform,
        noise_transform=None
    )

    test_ds = EnhancedMultiTaskDataset(
        hf_datasets, test_metadata, TOKENIZER,
        base_transform=val_transform,
        noise_transform=None
    )
    
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Model with all new features
    model = CompleteMultiTaskAuditor(
        num_classes=NUM_CLASSES, 
        vocab_size=len(TOKENIZER.word_to_idx)
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    from math import inf
    best_val_loss = inf
    
    for epoch in range(EPOCHS):
        avg_losses = train_epoch(model, train_loader, optimizer, epoch + 1)
        val_acc, val_loss = evaluate(model, val_loader)
        
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"  Total Loss: {avg_losses['total']:.4f}")
        print(f"  Binary: {avg_losses['binary']:.4f} | Class: {avg_losses['class']:.4f}")
        print(f"  Seam: {avg_losses['seam']:.4f} | Rel.Adv: {avg_losses['relative_adv']:.4f}")
        print(f"  Faith: {avg_losses['faithfulness']:.4f} | Quality: {avg_losses['quality']:.4f}")
        print(f"  Val Acc: {val_acc * 100:.2f}% | Val Loss (Binary): {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/complete_auditor_best.pth")
            print(f"  [+] Saved new best model (Val Loss: {val_loss:.4f})")
            
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/complete_auditor_latest.pth")
    
    # Load best model for analysis
    print("\n" + "="*80)
    print("LOADING BEST MODEL")
    print("="*80)
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/complete_auditor_best.pth"))
    print("[+] Loaded best model checkpoint")

    # =========================================================================
    # CONFUSION MATRIX ON HELD-OUT TEST SET
    # =========================================================================
    print("\n" + "="*80)
    print("TEST SET EVALUATION + CONFUSION MATRIX")
    print("="*80)
    import itertools

    model.eval()
    all_true_binary, all_pred_binary = [], []
    all_true_class,  all_pred_class  = [], []

    with torch.no_grad():
        for batch in test_loader:
            images      = batch['image'].to(DEVICE)
            b_labels    = batch['binary_label'].to(DEVICE)
            c_labels    = batch['class_label'].to(DEVICE)
            text_tokens = batch['text_tokens'].to(DEVICE)
            timesteps   = batch['timestep'].to(DEVICE)

            outputs = model(images, text_tokens=text_tokens, timestep=timesteps)
            probs   = torch.sigmoid(outputs['binary_logits']).squeeze(1)
            preds_b = (probs > 0.5).long()
            preds_c = torch.argmax(outputs['class_logits'], dim=1)

            all_true_binary.extend(b_labels.long().cpu().tolist())
            all_pred_binary.extend(preds_b.cpu().tolist())
            all_true_class.extend(c_labels.cpu().tolist())
            all_pred_class.extend(preds_c.cpu().tolist())

    # ── Binary confusion matrix ──────────────────────────────────────────────
    binary_labels_list = ['Safe', 'Unsafe']
    binary_cm = [[0, 0], [0, 0]]
    for t, p in zip(all_true_binary, all_pred_binary):
        binary_cm[t][p] += 1
    test_acc_bin = sum(binary_cm[i][i] for i in range(2)) / max(len(all_true_binary), 1)
    print(f"\nBinary Test Accuracy: {test_acc_bin * 100:.2f}%")
    print("\nBinary Confusion Matrix (rows=True, cols=Pred):")
    print(f"{'':>10}" + "".join(f"{l:>10}" for l in binary_labels_list))
    for i, rl in enumerate(binary_labels_list):
        print(f"{rl:>10}" + "".join(f"{binary_cm[i][j]:>10}" for j in range(2)))

    # ── 4-class confusion matrix ─────────────────────────────────────────────
    num_cls = len(CLASS_NAMES)
    cm = [[0] * num_cls for _ in range(num_cls)]
    for t, p in zip(all_true_class, all_pred_class):
        cm[t][p] += 1
    test_acc_cls = sum(cm[i][i] for i in range(num_cls)) / max(len(all_true_class), 1)
    print(f"\n{num_cls}-Class Test Accuracy: {test_acc_cls * 100:.2f}%")
    print(f"\n{num_cls}-Class Confusion Matrix (rows=True, cols=Pred):")
    print(f"{'':>12}" + "".join(f"{n[:8]:>10}" for n in CLASS_NAMES))
    for i, rl in enumerate(CLASS_NAMES):
        print(f"{rl[:12]:>12}" + "".join(f"{cm[i][j]:>10}" for j in range(num_cls)))

    # ── Per-class Precision / Recall / F1 ────────────────────────────────────
    print("\nPer-Class Metrics (Test Set):")
    print(f"{'Class':>12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(num_cls)) - tp
        fn = sum(cm[i][j] for j in range(num_cls)) - tp
        prec    = tp / max(tp + fp, 1)
        rec     = tp / max(tp + fn, 1)
        f1      = 2 * prec * rec / max(prec + rec, 1e-8)
        support = sum(cm[i])
        print(f"{cls:>12} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {support:>10}")

    # ── Save confusion-matrix PNG ─────────────────────────────────────────────
    try:
        cm_array = np.array(cm)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        im = ax_cm.imshow(cm_array, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax_cm)
        ax_cm.set_xticks(range(num_cls))
        ax_cm.set_yticks(range(num_cls))
        ax_cm.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=11)
        ax_cm.set_yticklabels(CLASS_NAMES, fontsize=11)
        ax_cm.set_xlabel('Predicted Label', fontsize=12)
        ax_cm.set_ylabel('True Label', fontsize=12)
        ax_cm.set_title(f'{num_cls}-Class Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
        thresh = cm_array.max() / 2.0
        for i, j in itertools.product(range(num_cls), range(num_cls)):
            ax_cm.text(j, i, str(cm_array[i, j]), ha='center', va='center',
                       color='white' if cm_array[i, j] > thresh else 'black', fontsize=12)
        plt.tight_layout()
        cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_test.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close(fig_cm)
        print(f"\n[+] Confusion matrix saved to: {cm_path}")
    except Exception as e:
        print(f"[!] Could not save confusion matrix figure: {e}")


    # Analyze sample images with COMPLETE VISUALIZATION from TEST dataset
    print("\n" + "="*80)
    print("GENERATING COMPLETE VISUALIZATIONS (TEST SET)")
    print("="*80)

    num_samples = min(50, len(test_ds))

    for i in range(num_samples):
        sample = test_ds[i]
        meta   = test_metadata[i]

        ds_name = meta['ds']
        row_idx = meta['row']
        ds_row  = hf_datasets[ds_name][row_idx]
        original_image = ds_row['image']

        if isinstance(original_image, PILImage.Image):
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')

        print(f"\n[{i+1}/{num_samples}] Processing image {sample['idx']}...")

        result = analyze_image_complete(
            model=model,
            image=original_image,
            prompt=meta['prompt'],
            idx=sample['idx'],
            binary_label=sample['binary_label'].item(),
            class_label=sample['class_label'].item(),
            quality_score=sample['quality_score'].item()
        )

        if i < 5:  # Print first 5 summaries
            print(f"  Prompt: {result['prompt'][:60]}...")
            print(f"  Predicted: {result['predictions']['predicted_class']} ({result['predictions']['adversarial_probability']:.1%})")
            print(f"  Seam Quality: {result['predictions']['seam_quality']:.3f}")
            print(f"  Relative Adv: {result['predictions']['relative_adversary_score']:.3f}")
            print(f"  Faithfulness: {result['predictions']['faithfulness_score']:.3f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Combined visualizations saved to: {COMBINED_DIR}")
    print(f"Individual heatmaps saved to: {HEATMAP_DIR}")
    print(f"Outlined images saved to: {OUTLINED_DIR}")
    print(f"Metadata JSON saved to: {METADATA_DIR}")
    print("="*80)
    
    print("\n" + "="*80)
    print("SUMMARY OF FEATURES")
    print("="*80)
    print("  [+] Seam quality assessment (detects inpainting artifacts)")
    print("  [+] Relative adversary score (continuous 0-1)")
    print("  [+] Text-conditioned faithfulness (actual prompt processing)")
    print("  [+] Timestep-aware features (diffusion timestep embedding)")
    print("  [+] Complete combined visualizations")
    print("  [+] All heatmaps, scores, and explanations in single images")
    print("  [+] Individual heatmap exports")
    print("  [+] Object detection with outlines")
    print("="*80)

if __name__ == "__main__":
    main()