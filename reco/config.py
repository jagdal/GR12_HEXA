"""Chemins et réglages du modèle de recommandation."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = Path(os.getenv("RECO_RAW_DIR", BASE_DIR / "data" / "row"))
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "data" / "eval"
CACHE_DIR = BASE_DIR / "data" / "cache"
MODEL_PATH = BASE_DIR / "models" / "modele_reco.json"
REPORTS_DIR = BASE_DIR / "reports"

# Les 5 axes, dans cet ordre partout (curseurs, profils, distances)
AXES = ["chaleur", "fraicheur", "floral", "boise", "intensite"]

# LLM retenu par le benchmark de l'équipe (benchmark_llm.pdf)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RECO_LLM_MODEL = os.getenv("RECO_LLM_MODEL", "qwen2.5:3b")

# Taux de change indicatifs pour comparer les prix en euros.
# À mettre à jour avant toute utilisation commerciale.
TAUX_EUR = {"EUR": 1.0, "USD": 0.86, "GBP": 1.15}
TAUX_DATE = "2026-11-12 (valeurs approximatives)"

# Seed unique pour toute partie aléatoire (découpage train/test)
SEED = 42
