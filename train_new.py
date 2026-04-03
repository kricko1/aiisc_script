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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
EPOCHS = 10
NUM_CLASSES = 5
NOISE_TIMESTEPS = 1000
CAP = 400
MAX_PROMPT_LENGTH = 77 

# Output directories
OUTPUT_DIR = "./analysis_outputs"
HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps")
OUTLINED_DIR = os.path.join(OUTPUT_DIR, "outlined_images")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")
COMBINED_DIR = os.path.join(OUTPUT_DIR, "combined_visualizations")
TEST_RAW_IMAGES_DIR = os.path.join(OUTPUT_DIR, "test_raw_images")
CHECKPOINT_DIR = "./checkpoints"

# =============================================================================
# SAFETY CATEGORIES
# =============================================================================

class SafetyCategories:
    CATEGORIES = ['safe', 'violence', 'sexual', 'illegal_activity', 'disturbing']
    NUM_CLASSES = 5
    IDX = {cat: i for i, cat in enumerate(CATEGORIES)}
    
CLASS_NAMES = ['Safe', 'Violence', 'Sexual', 'Illegal Activity', 'Disturbing']
CLASS_COLORS = [(100, 200, 100), (255, 100, 100), (200, 0, 0), (150, 150, 255), (180, 100, 200)]

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
    """ Load dataset from kricko/cleaned_auditor, apply balanced sampling 
    (1500 per class, excluding 'hate'), and split into train/val/test.
    """
    hf_datasets = {}
    
    print("\n" + "="*80)
    print("LOADING AND SAMPLING DATASETS")
    print("="*80)

    # Column name in the HF dataset → SafetyCategories key
    COL_TO_CAT = {
        'safe':             'safe',
        'violence':         'violence',
        'sexual':           'sexual',
        'illegal activity': 'illegal_activity',
        'disturbing':       'disturbing',
    }

    try:
        print("\n[1/1] Loading kricko/cleaned_auditor...")
        # Force redownload to avoid schema mismatch with cached metadata
        ds = load_dataset(
            "kricko/cleaned_auditor", 
            split="train", 
            download_mode="force_redownload"
        )
        hf_datasets['cleaned_auditor'] = ds
        print(f"  Loaded {len(ds)} rows. Columns: {ds.column_names}")

        label_cols = [c for c in COL_TO_CAT.keys() if c in ds.column_names]
        cols_needed = label_cols + (['prompt'] if 'prompt' in ds.column_names else [])
        
        # We'll group indices by their active class
        class_indices = {cat: [] for cat in COL_TO_CAT.values()}
        
        # To avoid massive memory usage if we only need a few thousand, 
        # we can still load all metadata or just iterate.
        print("  Scanning dataset for class distribution...")
        df = ds.select_columns(cols_needed).to_pandas()
        
        for j, row in df.iterrows():
            active_cats = [
                COL_TO_CAT[col] for col in label_cols
                if row.get(col, 0) == 1
            ]
            if not active_cats:
                active_cats.append('safe')
            
            # For sampling, assign to the FIRST active category matching our target list
            # Priorities: safe (last), then others
            primary_cat = active_cats[0]
            if primary_cat in class_indices:
                class_indices[primary_cat].append(j)

        # Labels to sample (1500 each)
        target_labels = ['disturbing', 'illegal_activity', 'safe', 'sexual', 'violence']
        sampled_metadata = []
        
        print("\n  Balanced Sampling (max 1500 per class):")
        random.seed(42) # For reproducibility
        
        for cat in target_labels:
            indices = class_indices[cat]
            if not indices:
                print(f"    [!] No samples found for {cat}")
                continue
                
            num_to_sample = min(1500, len(indices))
            sampled_idx = random.sample(indices, num_to_sample)
            print(f"    {cat:20s}: {len(indices):>6} available -> {num_to_sample:>4} sampled")
            
            for idx in sampled_idx:
                row = df.iloc[idx]
                active_cats = [
                    COL_TO_CAT[col] for col in label_cols
                    if row.get(col, 0) == 1
                ]
                # Re-calculate safety vector for this row
                sv = make_safety_vector(active_cats)
                binary_label = 1 if any(c != 'safe' for c in active_cats) else 0
                prompt_str = str(row.get('prompt', ''))
                
                sampled_metadata.append({
                    'ds':           'cleaned_auditor',
                    'row':          int(idx),
                    'safety_vec':   sv,
                    'binary_label': binary_label,
                    'prompt':       prompt_str,
                })

    except Exception as e:
        print(f"  [!] Failed to load dataset: {e}")
        raise

    if not sampled_metadata:
        raise ValueError("No metadata sampled!")

    random.shuffle(sampled_metadata)
    
    # Split: 70% Train, 15% Val, 15% Test
    total = len(sampled_metadata)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)
    
    train_metadata = sampled_metadata[:train_end]
    val_metadata   = sampled_metadata[train_end:val_end]
    test_metadata  = sampled_metadata[val_end:]

    # Build tokenizer vocabulary from ALL prompts in the sampled set
    print("\n" + "="*80)
    print("BUILDING TEXT TOKENIZER")
    print("="*80)
    all_prompts = [m['prompt'] for m in sampled_metadata if m['prompt']]
    TOKENIZER.build_vocab(all_prompts)
    print(f"[+] Vocabulary size: {len(TOKENIZER.word_to_idx)}")
    
    # Save vocabulary for inference
    vocab_path = os.path.join(CHECKPOINT_DIR, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(TOKENIZER.word_to_idx, f)
    print(f"[+] Saved vocabulary to {vocab_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("DATASET STRATIFIED SPLIT SUMMARY")
    print("="*80)
    print(f"Total Sampled : {len(sampled_metadata)}")
    print(f"  Train       : {len(train_metadata)}")
    print(f"  Val         : {len(val_metadata)}")
    print(f"  Test        : {len(test_metadata)}")
    
    print("\nPer-Class (Combined):")
    for i, cat in enumerate(SafetyCategories.CATEGORIES):
        count = sum(1 for m in sampled_metadata if m['safety_vec'][i] > 0)
        print(f"  {cat:20s}: {count}")
    print("="*80)

    gc.collect()
    return hf_datasets, train_metadata, val_metadata, test_metadata

# =============================================================================
# DIFFUSION NOISE AUGMENTATION
# =============================================================================

class DiffusionNoiseAugment:
    """Simulates diffusion noise at specific timesteps (last 20 to 30% of process)"""
    def __init__(self, max_steps=1000, p=0.5):
        self.max_steps = max_steps
        self.p = p
        self.betas = torch.linspace(0.0001, 0.02, max_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def __call__(self, img_tensor):
        if random.random() > self.p:
            return img_tensor, 0 

        # Noise pertaining to last 20 to 30 % steps of the diffusion process (e.g. 700 to 800)
        start_t = int(self.max_steps * 0.7)
        end_t = int(self.max_steps * 0.8)
        t = random.randint(start_t, end_t)
        noise = torch.randn_like(img_tensor)
        
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alphas_cumprod[t])
        
        noisy_img = sqrt_alpha_bar * img_tensor + sqrt_one_minus_alpha_bar * noise
        return noisy_img, t

# =============================================================================
# TEXT ENCODER FOR PROMPT CONDITIONING
# =============================================================================

class SimpleTextEncoder(nn.Module):
    """Word-embedding BiLSTM text encoder with per-token fc projection."""
    def __init__(self, vocab_size=50000, embed_dim=512, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 512)
        self.norm = nn.LayerNorm(512)   # pre-LN applied to seq_features before xattn
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_tokens):
        """
        Args:
            text_tokens: [B, seq_len] long
        Returns:
            text_features:  [B, 512]           – global CLS-style feature
            seq_features:   [B, seq_len, 512]  – per-token (pre-LN), used as K/V in xattn
            padding_mask:   [B, seq_len] bool  – True where token == PAD (ignored in attn)
        """
        B = text_tokens.size(0)
        device = text_tokens.device

        if B == 0:
            return (torch.zeros(1, 512, device=device),
                    torch.zeros(1, 1, 512, device=device),
                    torch.zeros(1, 1, dtype=torch.bool, device=device))

        padding_mask = (text_tokens == 0)                           # [B, L], True = PAD
        embedded = self.dropout(self.embedding(text_tokens))        # [B, L, 512]
        out, (hidden, _) = self.lstm(embedded)                      # out: [B, L, 512]
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)           # [B, 512]
        text_features = self.fc(hidden)                             # [B, 512]
        # Project per-token states and apply pre-LN for stable xattn K/V
        seq_features = self.norm(self.fc(out))                      # [B, L, 512]
        return text_features, seq_features, padding_mask


# =============================================================================
# COMPLETE MULTI-TASK RESNET101 MODEL
# =============================================================================

class CompleteMultiTaskAuditor(nn.Module):
    """
    ResNet101 multi-task adversarial image auditor.

    Tasks
    -----
    1. Binary adversarial detection              (adv_head)
    2. Safety category classification            (class_head)
    3. Quality regression                        (quality_head)
    4. Per-class object heatmaps                 (object_detection_head)
    5. CLIP-style image-prompt alignment         (img_proj_head / txt_proj_head)
    6. Seam / inpainting artifact detection      (seam_quality_head, FiLM-conditioned)
    7. Relative adversary score (continuous)     (relative_adv_head, FiLM-conditioned)
    8. Timestep-aware diffusion analysis         (timestep_embed → FiLM)

    Architecture improvements (v2)
    ──────────────────────────────
    - Cross-attention uses pre-LayerNorm on Q and K/V + key_padding_mask.
    - Faithfulness = CLIP contrastive, NOT BCE on safety labels.
    - Timestep embedding fused via FiLM (γ·x + β) into relative_adv and seam heads.
    - Decoupled loss objectives (safety labels never reach text encoder).
    """
    def __init__(self, num_classes=6, vocab_size=50000):
        super().__init__()
        print("\n" + "="*80)
        print("INITIALIZING COMPLETE AUDITOR MODEL (v2 — research sound)")
        print("="*80)

        # ── Backbone ─────────────────────────────────────────────────────────
        print("Loading ResNet101 backbone...")
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-2])   # [B, 2048, H, W]

        # ── Text encoder ─────────────────────────────────────────────────────
        print("Initializing Text Encoder...")
        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=512, hidden_dim=256)

        # ── Safety heads (supervised by safety labels only) ───────────────────
        print("Initializing Safety Heads...")
        self.adv_head    = nn.Conv2d(2048, 1, kernel_size=1)
        self.class_head  = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.quality_head = nn.Conv2d(2048, 1, kernel_size=1)
        self.object_detection_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        # ── Cross-attention path (image queries text tokens) ──────────────────
        print("  [+] Cross-Attention (pre-LN, key_padding_mask)")
        self.image_proj   = nn.Conv2d(2048, 512, kernel_size=1)
        self.query_norm   = nn.LayerNorm(512)   # pre-LN on Q (image patches)
        self.key_norm     = nn.LayerNorm(512)   # pre-LN on K/V (text tokens)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True, dropout=0.1
        )

        # ── CLIP-style faithfulness projection heads ──────────────────────────
        print("  [+] CLIP Faithfulness Projection Heads")
        self.img_proj_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.txt_proj_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        # Learnable log-temperature for InfoNCE (init ≈ 0.07)
        self.log_temperature = nn.Parameter(torch.tensor([-2.659]))   # ln(0.07)

        # ── Timestep embedding ────────────────────────────────────────────────
        print("  [+] Timestep Embedding + FiLM Heads")
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 512)
        )
        # FiLM for relative adversary head: modulates the 2048-d global feature
        self.film_adv  = nn.Linear(512, 2048 * 2)   # → (gamma [2048], beta [2048])
        # FiLM for seam quality head: modulates the 512-d first-conv feature
        self.film_seam = nn.Linear(512, 512 * 2)    # → (gamma [512], beta [512])

        # ── Relative adversary head (FiLM-conditioned) ───────────────────────
        print("  [+] Relative Adversary Score Head")
        self.relative_adv_head = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

        # ── Seam quality head (FiLM-conditioned) ─────────────────────────────
        print("  [+] Seam Quality Assessment Head")
        # Split into a feature extractor (FiLM applied between convs) and classifier
        self.seam_feat = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.seam_cls = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, kernel_size=1)
        )
        print("="*80)

    def forward(self, x, text_tokens=None, timestep=None, return_features=False):
        """
        Args:
            x:            [B, 3, 224, 224]   input images
            text_tokens:  [B, seq_len] long  tokenized prompts (PAD=0)
            timestep:     [B, 1] float       normalized diffusion timestep (0-1)
            return_features: bool            whether to also return raw feature maps
        """
        B = x.size(0)
        feats       = self.features(x)                                  # [B, 2048, H, W]
        global_feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)  # [B, 2048]

        # ── Safety heads ────────────────────────────────────────────────────
        adv_map     = self.adv_head(feats)
        adv_logits  = F.adaptive_avg_pool2d(adv_map, (1, 1)).flatten(1)         # [B, 1]
        class_map   = self.class_head(feats)
        class_logits = F.adaptive_avg_pool2d(class_map, (1, 1)).flatten(1)      # [B, C]
        qual_map    = self.quality_head(feats)
        qual_logits = F.adaptive_avg_pool2d(qual_map, (1, 1)).flatten(1)        # [B, 1]
        object_heatmaps = self.object_detection_head(feats)                     # [B, C, H, W]

        # ── Text encoding ────────────────────────────────────────────────────
        if text_tokens is not None:
            text_features, seq_features, padding_mask = self.text_encoder(text_tokens)
        else:
            text_features = torch.zeros(B, 512, device=x.device)
            seq_features  = torch.zeros(B, 1, 512, device=x.device)
            padding_mask  = torch.zeros(B, 1, dtype=torch.bool, device=x.device)

        # ── Cross-attention: image patches ← text tokens ─────────────────────
        img_feats_proj  = self.image_proj(feats)                         # [B, 512, H, W]
        Bi, Ci, Hi, Wi  = img_feats_proj.shape
        img_seq         = img_feats_proj.view(Bi, Ci, -1).permute(0, 2, 1)  # [B, H*W, 512]

        # Pre-LayerNorm on Q and K/V before attention
        img_seq_normed  = self.query_norm(img_seq)                       # [B, H*W, 512]
        seq_feat_normed = self.key_norm(seq_features)                    # [B, L, 512]

        attended_img_seq, _ = self.cross_attention(
            img_seq_normed,
            seq_feat_normed,
            seq_feat_normed,
            key_padding_mask=padding_mask   # True positions are ignored
        )                                                                # [B, H*W, 512]
        attended_img_feat = attended_img_seq.mean(dim=1)                 # [B, 512]

        # ── CLIP-style faithfulness embeddings ───────────────────────────────
        img_embed = F.normalize(self.img_proj_head(attended_img_feat), dim=-1)  # [B, 256]
        txt_embed = F.normalize(self.txt_proj_head(text_features),     dim=-1)  # [B, 256]

        # ── Timestep embedding + FiLM ────────────────────────────────────────
        timestep_features = None
        if timestep is not None:
            ts_feat = self.timestep_embed(timestep)                      # [B, 512]
            timestep_features = ts_feat

            # FiLM for relative adversary: modulate global image feature
            gbeta_adv = self.film_adv(ts_feat)                           # [B, 4096]
            gamma_adv, beta_adv = gbeta_adv.chunk(2, dim=-1)            # each [B, 2048]
            global_feats_mod = (1.0 + gamma_adv) * global_feats + beta_adv
        else:
            global_feats_mod = global_feats

        # ── Relative adversary head (FiLM-conditioned) ───────────────────────
        relative_adv_score = torch.sigmoid(
            self.relative_adv_head(global_feats_mod)                    # [B, 1]
        )

        # ── Seam quality head (FiLM-conditioned between conv stages) ─────────
        seam_mid = self.seam_feat(feats)                                 # [B, 512, H, W]
        if timestep is not None:
            gamma_seam, beta_seam = self.film_seam(ts_feat).chunk(2, dim=-1)  # each [B, 512]
            # Broadcast spatial dims
            seam_mid = (1.0 + gamma_seam[:, :, None, None]) * seam_mid \
                          + beta_seam[:, :, None, None]
        # Applying sigmoid directly to the map instead of after averaging
        # This makes the map values mathematically coherent probabilities [0, 1]
        seam_quality_map   = torch.sigmoid(self.seam_cls(seam_mid))     # [B, 1, H, W]
        seam_quality_score = F.adaptive_avg_pool2d(seam_quality_map, (1, 1)).flatten(1)  # [B, 1]

        outputs = {
            # ── Safety outputs (supervised by safety labels only) ──
            'binary_logits':     adv_logits,        # [B, 1]
            'class_logits':      class_logits,      # [B, C]
            'quality_logits':    qual_logits,       # [B, 1]
            'object_heatmaps':   object_heatmaps,   # [B, C, H, W]
            'adversarial_map':   adv_map,           # [B, 1, H, W]
            'class_map':         class_map,         # [B, C, H, W]
            # ── Alignment outputs (supervised by contrastive loss only) ──
            'img_embed':         img_embed,         # [B, 256] L2-normalized
            'txt_embed':         txt_embed,         # [B, 256] L2-normalized
            'attended_img_feat': attended_img_feat, # [B, 512]
            'text_features':     text_features,     # [B, 512]
            # ── Auxiliary quality outputs ──
            'seam_quality_map':     seam_quality_map,    # [B, 1, H, W]
            'seam_quality_score':   seam_quality_score,  # [B, 1] 0-1
            'relative_adv_score':   relative_adv_score,  # [B, 1] 0-1
            'timestep_features':    timestep_features,   # [B, 512] or None
        }

        if return_features:
            outputs['features']        = feats
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
        
        #  Seam quality proxy: lower if image is adversarial, and heavily penalized by diffusion noise
        seam_val = 1.0 - (0.2 * binary_label) - (0.8 * normalized_timestep)
        seam_quality_gt = torch.tensor(seam_val, dtype=torch.float32).clamp(0.0, 1.0)

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
# CONTRASTIVE ALIGNMENT LOSS  (InfoNCE / CLIP-style)
# =============================================================================

def info_nce_loss(img_embed, txt_embed, log_temperature):
    """
    Symmetric InfoNCE loss (NT-Xent) for image-text alignment.

    Positive pairs : (img_i, txt_i)  — same index in batch
    Negative pairs : all off-diagonal cross-modal pairs within the batch

    Args:
        img_embed:       [B, D] L2-normalized image embeddings
        txt_embed:       [B, D] L2-normalized text embeddings
        log_temperature: scalar learnable parameter  (log scale for stability)

    Returns:
        Scalar contrastive loss (mean of image→text and text→image directions)

    Why InfoNCE and not BCEWithLogitsLoss?
    ─────────────────────────────────────
    BCE with binary labels forces faithfulness to predict a *supervised* safe/unsafe
    signal, making it a redundant copy of the adversarial head.  InfoNCE compares
    each image against all other captions in the batch: the model must push the
    correct (img, txt) pair together while separating ALL other cross-modal pairs,
    which provides real prompt-image alignment signal independent of safety labels.
    """
    B = img_embed.size(0)
    if B < 2:
        # Can't compute contrastive loss with a single sample
        return img_embed.sum() * 0.0

    temperature = log_temperature.exp().clamp(min=1e-4, max=1.0)
    # Similarity matrix: [B, B], sim[i,j] = cosine sim of img_i and txt_j
    sim = (img_embed @ txt_embed.T) / temperature              # [B, B]
    labels = torch.arange(B, device=img_embed.device)         # diagonal = positives
    loss_i2t = F.cross_entropy(sim,   labels)                 # image → text
    loss_t2i = F.cross_entropy(sim.T, labels)                 # text  → image
    return (loss_i2t + loss_t2i) * 0.5


# =============================================================================
# TRAINING ENGINE
# =============================================================================

def train_epoch(model, loader, optimizer, epoch_num):
    """
    Train for one epoch.

    Loss structure (fully decoupled):
    ──────────────────────────────────────────────────────────────────────────
    L = 1.0 * L_binary          BCE   binary adversarial detection
      + 0.5 * L_class           CE    safety category classification
      + 0.3 * L_quality         MSE   image quality regression
      + 0.4 * L_relative_adv    MSE   continuous adversarial strength score
      + 0.3 * L_seam            MSE   seam/inpainting artifact score
      + 0.5 * L_contrastive     InfoNCE  image–prompt alignment (CLIP-style)

    Key invariant: safety labels (b_labels) NEVER touch the text encoder or
    the faithfulness/alignment branch.  The contrastive term is the ONLY
    gradient source for img_proj_head, txt_proj_head, and text_encoder.
    ──────────────────────────────────────────────────────────────────────────
    """
    model.train()
    total_loss = 0.0
    loss_components = {
        'binary': 0.0, 'class': 0.0, 'quality': 0.0,
        'relative_adv': 0.0, 'seam': 0.0, 'contrastive': 0.0
    }

    criterion_binary  = nn.BCEWithLogitsLoss()
    criterion_class   = nn.CrossEntropyLoss()
    criterion_qual    = nn.MSELoss()
    criterion_seam    = nn.MSELoss()
    criterion_rel_adv = nn.MSELoss()

    try:
        from tqdm import tqdm
        pbar = tqdm(loader, desc=f"Epoch {epoch_num}")
    except ImportError:
        pbar = loader
        print(f"Epoch {epoch_num} running...")

    for batch in pbar:
        images      = batch['image'].to(DEVICE)
        b_labels    = batch['binary_label'].to(DEVICE).unsqueeze(1)   # [B, 1]
        c_labels    = batch['class_label'].to(DEVICE)                 # [B]
        q_labels    = batch['quality_score'].to(DEVICE).unsqueeze(1)  # [B, 1]
        text_tokens = batch['text_tokens'].to(DEVICE)                 # [B, L]
        timesteps   = batch['timestep'].to(DEVICE)                    # [B, 1]
        seam_gt     = batch['seam_quality_gt'].to(DEVICE).unsqueeze(1) # [B, 1]

        optimizer.zero_grad()
        outputs = model(images, text_tokens=text_tokens, timestep=timesteps)

        # ── Safety losses (safety labels only reach safety heads) ─────────────
        loss_b        = criterion_binary(outputs['binary_logits'], b_labels)
        loss_c        = criterion_class(outputs['class_logits'], c_labels)
        loss_q        = criterion_qual(torch.sigmoid(outputs['quality_logits']), q_labels)
        loss_rel_adv  = criterion_rel_adv(outputs['relative_adv_score'], b_labels)
        loss_seam     = criterion_seam(outputs['seam_quality_score'], seam_gt)

        # ── Alignment loss (safety labels NEVER used here) ───────────────────
        # InfoNCE: each image must be closest to its own prompt, not any other.
        loss_contrastive = info_nce_loss(
            outputs['img_embed'],
            outputs['txt_embed'],
            model.log_temperature
        )

        # ── Combined λ-weighted loss ─────────────────────────────────────────
        loss = (
            1.0 * loss_b            +   # primary safety objective
            0.5 * loss_c            +   # category breakdown
            0.3 * loss_q            +   # quality (lighter)
            0.4 * loss_rel_adv      +   # continuous adversarial strength
            0.3 * loss_seam         +   # seam/artifact quality
            0.5 * loss_contrastive      # CLIP alignment (equal to class weight)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loss_components['binary']      += loss_b.item()
        loss_components['class']       += loss_c.item()
        loss_components['quality']     += loss_q.item()
        loss_components['relative_adv'] += loss_rel_adv.item()
        loss_components['seam']        += loss_seam.item()
        loss_components['contrastive'] += loss_contrastive.item()

        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'loss':  f'{loss.item():.4f}',
                'bin':   f'{loss_b.item():.3f}',
                'nce':   f'{loss_contrastive.item():.3f}',
                'seam':  f'{loss_seam.item():.3f}',
                'temp':  f'{model.log_temperature.exp().item():.3f}',
            })

    n = len(loader)
    avg_losses = {k: v / n for k, v in loss_components.items()}
    avg_losses['total'] = total_loss / n
    return avg_losses

def evaluate(model, loader):
    """Evaluate binary accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(DEVICE)
            b_labels = batch['binary_label'].to(DEVICE).unsqueeze(1)
            text_tokens = batch['text_tokens'].to(DEVICE)
            timesteps = batch['timestep'].to(DEVICE)
            
            outputs = model(images, text_tokens=text_tokens, timestep=timesteps)
            probs = torch.sigmoid(outputs['binary_logits'])
            preds = (probs > 0.5).float()
            
            total += b_labels.size(0)
            correct += (preds == b_labels).sum().item()
            
    return correct / total if total > 0 else 0.0


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def create_outlined_image(image, outputs, metadata):
    """
    Create outlined version with precise contours for detected objects
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
    widths = [3, 2, 1]
    
    for threshold, color, width in zip(thresholds, colors_by_threshold, widths):
        binary_mask = (heatmap_norm > threshold).astype(np.uint8) * 255
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small noise and draw precise contours
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        cv2.drawContours(img_np, valid_contours, -1, color, width)
        
    img_pil = PILImage.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    
    # Draw legend
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except:
        font = ImageFont.load_default()
    
    legend_y = 10
    draw.text((10, legend_y), f"{CLASS_NAMES[class_idx]} Detection:", fill=(255, 255, 255), font=font)
    legend_y += 15
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


def create_combined_visualization(model, image, prompt, idx, metadata, outputs, image_tensor,
                                  override_output_path=None):
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
    # Since we have 6 classes now, we need at least 6 columns, plus the first 4 standard plots, so 10+ columns.
    # Alternatively, 2 rows of 3 for the overlays.
    # Let's adjust GridSpec: 5 rows, 7 columns to give plenty of space
    fig = plt.figure(figsize=(30, 20))
    gs = GridSpec(6, max(7, NUM_CLASSES), figure=fig, hspace=0.4, wspace=0.35)
    
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
        # We put them across ROW 2, starting at column 4, and spilling into ROW 3 if needed
        row_idx = 1 if i < 3 else 2
        col_idx = 4 + (i % 3)
        ax = fig.add_subplot(gs[row_idx, col_idx])
        
        class_heatmap = outputs['object_heatmaps'][0, i].cpu().numpy()
        overlay = apply_heatmap_overlay(image, class_heatmap, colormap='hot', alpha=0.5)
        ax.imshow(overlay)
        ax.set_title(f'{CLASS_NAMES[i]}\nOverlay', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # ============= ROW 3: ALL CLASS DETECTION HEATMAPS (SIDE BY SIDE) =============
    
    for i in range(NUM_CLASSES):
        ax = fig.add_subplot(gs[3, i])
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
    ax_hist1 = fig.add_subplot(gs[2, 0:2])
    adv_flat = adv_map.flatten()
    ax_hist1.hist(adv_flat, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax_hist1.set_title('Adversarial\nScore Distribution', fontsize=11, fontweight='bold')
    ax_hist1.set_xlabel('Activation Value')
    ax_hist1.set_ylabel('Frequency')
    ax_hist1.grid(alpha=0.3)
    
    # Histogram of class detection
    ax_hist2 = fig.add_subplot(gs[2, 2:4])
    class_flat = class_map.flatten()
    ax_hist2.hist(class_flat, bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax_hist2.set_title(f'{CLASS_NAMES[class_idx]}\nScore Distribution', fontsize=11, fontweight='bold')
    ax_hist2.set_xlabel('Activation Value')
    ax_hist2.set_ylabel('Frequency')
    ax_hist2.grid(alpha=0.3)
    
    # ============= ROW 4: METRICS AND SCORES =============
    
    # 11. Main Scores Panel
    ax11 = fig.add_subplot(gs[4, 0:2])
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
    ax12 = fig.add_subplot(gs[4, 2:])
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
    
    ax13 = fig.add_subplot(gs[5, 0:4])
    
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
    ax14 = fig.add_subplot(gs[5, 4:])
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
    
    # Save combined visualization — use override path (per-label dir) if provided
    output_path = override_output_path if override_output_path else \
        os.path.join(COMBINED_DIR, f"complete_analysis_{idx:06d}.png")
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

def analyze_image_complete(model, image, prompt, idx, binary_label, class_label, quality_score,
                           output_subdir=None, test_viz_dir=None):
    """
    Complete analysis with ALL features including comprehensive visualization
    
    This function:
    1. Runs model inference
    2. Generates all predictions and scores
    3. Creates combined visualization with all heatmaps, scores, and explanations
    4. Saves individual heatmaps
    5. Saves metadata JSON
    
    output_subdir: when provided (e.g. class name), combined viz is saved into
                   test_viz_dir/<output_subdir>/ for per-label organization.
    
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
    faithfulness_score = (
        F.cosine_similarity(outputs['img_embed'], outputs['txt_embed'], dim=-1).item() + 1.0
    ) / 2.0  # cosine in [-1,1] → [0,1]; independent of safety labels
    

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
    # If a per-label subdir is specified, save the combined viz there directly
    if output_subdir and test_viz_dir:
        label_viz_path = os.path.join(test_viz_dir, output_subdir,
                                      f"complete_analysis_{idx:06d}.png")
        viz_path = create_combined_visualization(
            model, image, prompt, idx, metadata, outputs, image_tensor,
            override_output_path=label_viz_path
        )
    else:
        viz_path = create_combined_visualization(
            model, image, prompt, idx, metadata, outputs, image_tensor
        )
    
    #  ALSO SAVE INDIVIDUAL HEATMAPS (with overlays)
    save_individual_heatmaps(outputs, idx, image)
    
    # Save outlined image separately as well
    outlined_img = create_outlined_image(image, outputs, metadata)
    outlined_path = os.path.join(OUTLINED_DIR, f"test_outlined_{idx:06d}.png")
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

def inference(model, image, prompt=""):
    """
    Independent plug-and-play function for auditing an image.

    Args:
        model:  Trained CompleteMultiTaskAuditor model
        image:  PIL Image
        prompt: String text prompt used to generate the image

    Returns:
        dict:
            - global_safety_score:     float (1.0 = fully safe, 0.0 = adversarial)
            - category_probabilities:  dict of class name -> probability
            - faithfulness_score:      float [0, 1] — cosine sim of img_embed and
                                       txt_embed, normalized to [0, 1].  Measures
                                       image–prompt alignment independent of safety.
            - seam_quality:            float [0, 1] (1.0 = no artifacts)
            - binary_map:              2D np.ndarray — raw adversarial activation map
            - heatmap:                 2D np.ndarray [224, 224], normalized [0, 1] —
                                       class-specific detection heatmap for the
                                       predicted class (use apply_heatmap_overlay to
                                       visualize it on the original image)
            - predicted_class:         str — name of the predicted safety class
            - predicted_class_idx:     int — index of the predicted class in CLASS_NAMES
    """
    was_training = model.training
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if isinstance(image, PILImage.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')

    if len(TOKENIZER.word_to_idx) <= 4:
        print("[!] Warning: tokenizer vocabulary not built. Call load_datasets_lazy() first.")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    text_tokens  = TOKENIZER.encode(prompt).unsqueeze(0).to(DEVICE)
    timestep     = torch.tensor([[0.0]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor, text_tokens=text_tokens, timestep=timestep)

    binary_prob        = torch.sigmoid(outputs['binary_logits']).item()
    global_safety_score = 1.0 - binary_prob

    class_probs = F.softmax(outputs['class_logits'], dim=1)[0].cpu().numpy()
    category_probabilities = {CLASS_NAMES[i]: float(class_probs[i]) for i in range(len(CLASS_NAMES))}

    # Faithfulness = cosine similarity between L2-normalized embeddings, mapped to [0, 1]
    # This is INDEPENDENT of safety labels — purely image–prompt alignment.
    cos_sim = F.cosine_similarity(
        outputs['img_embed'], outputs['txt_embed'], dim=-1
    ).item()                                                      # in [-1, 1]
    faithfulness_score = (cos_sim + 1.0) / 2.0                   # re-scale to [0, 1]

    seam_quality = outputs['seam_quality_score'].item()
    binary_map   = outputs['adversarial_map'][0, 0].cpu().numpy()

    # Heatmap: class-specific object detection map for the predicted class,
    # upsampled to 224×224 and normalized to [0, 1].
    predicted_class_idx = int(np.argmax(class_probs))
    raw_heatmap = outputs['object_heatmaps'][0, predicted_class_idx].cpu().numpy()
    heatmap_224 = cv2.resize(raw_heatmap, (224, 224))
    heatmap_224 = (heatmap_224 - heatmap_224.min()) / (heatmap_224.max() - heatmap_224.min() + 1e-8)

    if was_training:
        model.train()

    return {
        "global_safety_score":    global_safety_score,
        "category_probabilities": category_probabilities,
        "faithfulness_score":     faithfulness_score,
        "seam_quality":           seam_quality,
        "binary_map":             binary_map,
        "heatmap":                heatmap_224,        # 2D np.ndarray [224,224], normalized [0,1]
        "predicted_class":        CLASS_NAMES[predicted_class_idx],
        "predicted_class_idx":    predicted_class_idx,
    }

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
    hf_datasets, train_metadata, val_metadata, test_metadata = load_datasets_lazy()
    
    # (The splitting is now handled within load_datasets_lazy)
    
    print(f"\n[+] Data splits prepared:")
    print(f"    Train samples: {len(train_metadata)}")
    print(f"    Val samples:   {len(val_metadata)}")
    print(f"    Test samples:  {len(test_metadata)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomAffine(degrees=0, shear=10),
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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
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
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        avg_losses = train_epoch(model, train_loader, optimizer, epoch + 1)
        val_acc = evaluate(model, val_loader)
        
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"  Total Loss:    {avg_losses['total']:.4f}")
        print(f"  Binary:        {avg_losses['binary']:.4f}  | Class: {avg_losses['class']:.4f}")
        print(f"  Seam:          {avg_losses['seam']:.4f}  | Rel.Adv: {avg_losses['relative_adv']:.4f}")
        print(f"  Contrastive:   {avg_losses['contrastive']:.4f}  | Quality: {avg_losses['quality']:.4f}")
        print(f"  Temperature:   {model.log_temperature.exp().item():.4f}")
        print(f"  Val Acc:       {val_acc * 100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/complete_auditor_best.pth")
            print(f"  [+] Saved new best model")
            
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/complete_auditor_latest.pth")
    
    # Load best model for analysis
    print("\n" + "="*80)
    print("LOADING BEST MODEL")
    print("="*80)
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/complete_auditor_best.pth"))
    print("[+] Loaded best model checkpoint")
    
    # Analyze ALL test images with COMPLETE VISUALIZATION from TEST dataset
    print("\n" + "="*80)
    print("GENERATING COMPLETE VISUALIZATIONS (FULL TEST SET)")
    print("="*80)
    
    num_samples = len(test_ds)  # Process ALL test images
    print(f"Processing all {num_samples} test images...")
    
    TEST_VIZ_DIR = "test_set_visualizations"
    os.makedirs(TEST_VIZ_DIR, exist_ok=True)
    # Create per-label subdirectories for the output visualizations
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(TEST_VIZ_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(TEST_RAW_IMAGES_DIR, class_name), exist_ok=True)
    
    test_label_counts = {name: 0 for name in CLASS_NAMES}
    all_true_labels = []
    all_pred_labels = []
    
    for i in range(num_samples):
        sample = test_ds[i]
        meta = test_metadata[i]
        
        ds_name = meta['ds']
        row_idx = meta['row']
        ds_row = hf_datasets[ds_name][row_idx]
        original_image = ds_row['image']
        
        if isinstance(original_image, PILImage.Image):
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
        
        # Determine true class label — used for output sub-directory routing
        actual_class = CLASS_NAMES[sample['class_label'].item()]
        test_label_counts[actual_class] += 1
        
        print(f"\n[{i+1}/{num_samples}] Processing image {sample['idx']} (True Label: {actual_class})...")
        
        result = analyze_image_complete(
            model=model,
            image=original_image,
            prompt=meta['prompt'],
            idx=sample['idx'],
            binary_label=sample['binary_label'].item(),
            class_label=sample['class_label'].item(),
            quality_score=sample['quality_score'].item(),
            output_subdir=actual_class,   # <-- save viz into correct label subdir
            test_viz_dir=TEST_VIZ_DIR
        )
        
        if i < 5:  # Print first 5 summaries
            print(f"  Prompt: {result['prompt'][:60]}...")
            print(f"  Predicted: {result['predictions']['predicted_class']} ({result['predictions']['adversarial_probability']:.1%})")
            print(f"  Seam Quality: {result['predictions']['seam_quality']:.3f}")
            print(f"  Relative Adv: {result['predictions']['relative_adversary_score']:.3f}")
            print(f"  Faithfulness: {result['predictions']['faithfulness_score']:.3f}")
            
        # Collect for metrics
        all_true_labels.append(actual_class)
        all_pred_labels.append(result['predictions']['predicted_class'])
        
        # Save raw image to label directory
        raw_path = os.path.join(TEST_RAW_IMAGES_DIR, actual_class, f"{sample['idx']}.jpg")
        original_image.save(raw_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Combined visualizations saved to: {COMBINED_DIR}")
    print(f"Individual heatmaps saved to: {HEATMAP_DIR}")
    print(f"Outlined images saved to: {OUTLINED_DIR}")
    print(f"Metadata JSON saved to: {METADATA_DIR}")
    print(f"Raw test images organized by label in: {TEST_RAW_IMAGES_DIR}")
    print("\nTest Set Label Distribution (Analyzed Samples):")
    for class_name, count in test_label_counts.items():
        if count > 0:
            print(f"  {class_name:15s}: {count} images")
    
    # ── Confusion Matrix ──
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRIX")
    print("="*80)
    
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=CLASS_NAMES)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix: Adversarial Image Auditor")
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"[+] Confusion matrix saved to: {cm_path}")
    
    # Classification Report
    report = classification_report(all_true_labels, all_pred_labels, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
    report_df.to_csv(report_path)
    print(f"[+] Classification report saved to: {report_path}")
    
    print("\nDetailed Metrics:")
    print(classification_report(all_true_labels, all_pred_labels, target_names=CLASS_NAMES))
    
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