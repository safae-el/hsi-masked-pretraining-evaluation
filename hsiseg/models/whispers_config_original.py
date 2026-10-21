"""
config.py
=========
Central configuration for all models.
Every model script imports from here — never hardcode paths or hyperparams.
"""

import os

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = "/mnt/c/Users/Safae/mamba-transformer-fusion"
DATA_DIR    = os.path.join(BASE_DIR, "data")
SPLITS_DIR  = os.path.join(BASE_DIR, "splits")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

HSI_PATH    = os.path.join(DATA_DIR, "Indian_pines_corrected.mat")
GT_PATH     = os.path.join(DATA_DIR, "Indian_pines_gt.mat")
SPLIT_PATH  = os.path.join(SPLITS_DIR, "split_seed42.npy")

# ── dataset ───────────────────────────────────────────────────────────────────
NUM_CLASSES = 16
IN_BANDS    = 200
IMG_SIZE    = 145
PATCH_SIZE  = 9

CLASS_NAMES = [
    "Alfalfa",
    "Corn-notill",
    "Corn-mintill",
    "Corn",
    "Grass-pasture",
    "Grass-trees",
    "Grass-pasture-mowed",
    "Hay-windrowed",
    "Oats",
    "Soybean-notill",
    "Soybean-mintill",
    "Soybean-clean",
    "Wheat",
    "Woods",
    "Buildings-Grass-Trees-Drives",
    "Stone-Steel-Towers",
]

# ── PaviaU paths ──────────────────────────────────────────────────────────────
PAVIA_HSI_PATH  = os.path.join(DATA_DIR, "PaviaU.mat")
PAVIA_GT_PATH   = os.path.join(DATA_DIR, "PaviaU_gt.mat")
PAVIA_SPLIT_PATH= os.path.join(SPLITS_DIR, "split_pavia_seed42.npy")
PAVIA_RESULTS_DIR = os.path.join(BASE_DIR, "results", "pavia")

# ── PaviaU dataset specs ──────────────────────────────────────────────────────
PAVIA_NUM_CLASSES = 9
PAVIA_IN_BANDS    = 103
PAVIA_IMG_H       = 610
PAVIA_IMG_W       = 340
PAVIA_PATCH_SIZE  = 9

PAVIA_CLASS_NAMES = [
    "Asphalt",
    "Meadows",
    "Gravel",
    "Trees",
    "Painted metal sheets",
    "Bare soil",
    "Bitumen",
    "Self-blocking bricks",
    "Shadows",
]

# ── Salinas paths ─────────────────────────────────────────────────────────────
SALINAS_HSI_PATH   = os.path.join(DATA_DIR, "Salinas_corrected.mat")
SALINAS_GT_PATH    = os.path.join(DATA_DIR, "Salinas_gt.mat")
SALINAS_SPLIT_PATH = os.path.join(SPLITS_DIR, "split_salinas_seed42.npy")
SALINAS_RESULTS_DIR = os.path.join(BASE_DIR, "results", "salinas")

# ── Salinas dataset specs ──────────────────────────────────────────────────────
SALINAS_NUM_CLASSES = 16
SALINAS_IN_BANDS    = 204
SALINAS_IMG_H       = 512
SALINAS_IMG_W       = 217
SALINAS_PATCH_SIZE  = 9

SALINAS_CLASS_NAMES = [
    "Brocoli_green_weeds_1",
    "Brocoli_green_weeds_2",
    "Fallow",
    "Fallow_rough_plow",
    "Fallow_smooth",
    "Stubble",
    "Celery",
    "Grapes_untrained",
    "Soil_vinyard_develop",
    "Corn_senesced_green_weeds",
    "Lettuce_romaine_4wk",
    "Lettuce_romaine_5wk",
    "Lettuce_romaine_6wk",
    "Lettuce_romaine_7wk",
    "Vinyard_untrained",
    "Vinyard_vertical_trellis",
]
# ── training — same for ALL models ────────────────────────────────────────────
SEED        = 42
TRAIN_RATIO = 0.10
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 1e-3
WEIGHT_DECAY= 1e-4

# ── paper reporting ───────────────────────────────────────────────────────────
PAPER_NOTE = {
    "dataset":    "Indian Pines (corrected, 200 bands, 145x145, 16 classes)",
    "split":      "Stratified 10% train / 90% test, seed=42",
    "optimizer":  "Adam, lr=1e-3, weight_decay=1e-4",
    "scheduler":  "CosineAnnealingLR",
    "epochs":     EPOCHS,
    "patch_size": PATCH_SIZE,
    "batch_size": BATCH_SIZE,
}
