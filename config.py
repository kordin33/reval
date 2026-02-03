"""
Centralna konfiguracja projektu AI Text Detection Academic.
Wszystkie thresholds, API endpoints i ustawienia w jednym miejscu.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional


BASE_DIR = Path(__file__).parent
CHUNKS_DIR = BASE_DIR / "chunks"
RESULTS_DIR = BASE_DIR / "results"
SAMPLES_DIR = BASE_DIR / "samples_for_testing"

RESULTS_DIR.mkdir(exist_ok=True)


# --- Thresholds ---

@dataclass
class Thresholds:
    very_safe: float = 20.0      # < 20% AI
    safe: float = 40.0           # 20-40%
    attention: float = 60.0      # 40-60%
    high_risk: float = 80.0      # 60-80%
    # > 80% = critical

    def classify(self, ai_pct: float) -> str:
        if ai_pct < 0:
            return "error"
        if ai_pct < self.very_safe:
            return "very_safe"
        if ai_pct < self.safe:
            return "safe"
        if ai_pct < self.attention:
            return "attention"
        if ai_pct < self.high_risk:
            return "high_risk"
        return "critical"

    def emoji(self, ai_pct: float) -> str:
        labels = {
            "very_safe": "✅",
            "safe": "🟢",
            "attention": "⚠️",
            "high_risk": "🔶",
            "critical": "🚨",
            "error": "❌",
        }
        return labels[self.classify(ai_pct)]

    def label_pl(self, ai_pct: float) -> str:
        labels = {
            "very_safe": "Bardzo bezpieczne",
            "safe": "Raczej bezpieczne",
            "attention": "Wymaga uwagi",
            "high_risk": "Wysokie ryzyko",
            "critical": "Krytyczne ryzyko",
            "error": "Blad",
        }
        return labels[self.classify(ai_pct)]


THRESHOLDS = Thresholds()


# --- API keys (z env vars) ---

@dataclass
class APIKeys:
    sapling: Optional[str] = field(default_factory=lambda: os.getenv("SAPLING_API_KEY"))
    gptzero: Optional[str] = field(default_factory=lambda: os.getenv("GPTZERO_API_KEY"))
    originality: Optional[str] = field(default_factory=lambda: os.getenv("ORIGINALITY_API_KEY"))
    copyleaks_key: Optional[str] = field(default_factory=lambda: os.getenv("COPYLEAKS_API_KEY"))
    copyleaks_email: Optional[str] = field(default_factory=lambda: os.getenv("COPYLEAKS_EMAIL"))
    huggingface: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))


API_KEYS = APIKeys()


# --- API Endpoints ---

API_ENDPOINTS = {
    "zerogpt": "https://api.zerogpt.com/api/detect/detectText",
    "sapling": "https://api.sapling.ai/api/v1/aidetect",
    "gptzero": "https://api.gptzero.me/v2/predict/text",
    "originality": "https://api.originality.ai/api/v2/scan/ai",
    "grammarly": "https://api.grammarly.com/ecosystem/api/v1/ai-detection",
    "copyleaks_auth": "https://id.copyleaks.com/v3/account/login/api",
    "copyleaks_detect": "https://api.copyleaks.com/v2/writer-detector",
    "contentdetector": "https://contentdetector.ai/api/detectAi",
}


# --- Detektor weights (do ensemble scoring) ---

DETECTOR_WEIGHTS: Dict[str, float] = {
    "Pangram-EditLens": 0.25,  # ICLR 2026, open-source, F1=1.0, najnowszy
    "Binoculars": 0.20,        # language-agnostic, zero-shot, najlepszy dla polskiego
    "Grammarly": 0.15,         # #1 na RAID benchmark 2026
    "GPTZero": 0.15,           # 99% accuracy, dobry na angielskim
    "ZeroGPT": 0.10,           # darmowy, dzialajacy bez klucza
    "Originality.ai": 0.10,    # platny ale dokladny
    "Sapling": 0.05,           # sentence-level
}


# --- JSA Analysis ---

FORMAL_PATTERNS_PL = [
    (r"z punktu widzenia", "Formalna fraza: 'z punktu widzenia'"),
    (r"nale[zż]y podkre[sś]li[cć]", "Formalna fraza: 'nalezy podkreslic'"),
    (r"w niniejszej pracy", "Formalna fraza: 'w niniejszej pracy'"),
    (r"podsumowuj[aą]c,?\s*mo[zż]na", "Formalna fraza: 'podsumowujac mozna'"),
    (r"w zwi[aą]zku z powy[zż]szym", "Formalna fraza: 'w zwiazku z powyzszym'"),
    (r"maj[aą]c na uwadze", "Formalna fraza: 'majac na uwadze'"),
    (r"istotnym aspektem jest", "Formalna fraza: 'istotnym aspektem jest'"),
    (r"zagadnienie to", "Formalna fraza: 'zagadnienie to'"),
    (r"powy[zż]sze rozwa[zż]ania", "Formalna fraza: 'powyzsze rozwazania'"),
    (r"w kontek[sś]cie", "Formalna fraza: 'w kontekscie'"),
    (r"na podstawie przeprowadzon", "Formalna fraza: 'na podstawie przeprowadzon...'"),
    (r"analiza wykazuje", "Formalna fraza: 'analiza wykazuje'"),
    (r"wyniki wskazuj[aą]", "Formalna fraza: 'wyniki wskazuja'"),
    (r"mo[zż]na stwierdzi[cć]", "Formalna fraza: 'mozna stwierdzic'"),
]


# --- Binoculars config ---

@dataclass
class BinocularsConfig:
    # Modele dobrane do 8GB VRAM (RTX 4060 Ti)
    observer_model: str = "meta-llama/Llama-3.2-1B"
    performer_model: str = "meta-llama/Llama-3.2-3B"
    # Threshold z oryginalnego paperu
    threshold_low_fpr: float = 0.9015  # ~0.01% FPR
    threshold_balanced: float = 0.8536  # balans precision/recall
    max_length: int = 512
    device: str = "auto"  # auto-detect GPU


BINOCULARS_CONFIG = BinocularsConfig()


# --- LaTeX cleanup patterns ---

LATEX_NOISE_PATTERNS = [
    "[KOD ŹRÓDŁOWY POMINIĘTY]",
    "[KOD ZRODLOWY POMINIETY]",
    "[STRUKTURA KATALOGÓW]",
    "[STRUKTURA KATALOGOW]",
    "[TABELA]",
    "{}",
    "{ }",
    "{\\ }",
]


# --- Sentence analysis ---

MIN_SENTENCE_LENGTH = 10       # min znakow zeby liczyc zdanie
MAX_GOOD_SENTENCE_WORDS = 25   # ponad to = za dlugie
MIN_GOOD_VARIANCE = 20.0       # ponizej = zbyt regularne
MIN_GOOD_UNIQUE_RATIO = 0.4    # ponizej = formulaiczny
