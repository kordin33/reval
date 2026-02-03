"""
Wspolne narzedzia dla projektu AI Text Detection Academic.
Eliminuje duplikacje kodu miedzy skryptami.
"""

import json
import time
import logging
import functools
import requests
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field, asdict

from config import (
    CHUNKS_DIR, RESULTS_DIR, LATEX_NOISE_PATTERNS,
    THRESHOLDS, API_ENDPOINTS,
)


# --- Logging ---

def setup_logging(name: str = "ai_detection", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    log_path = RESULTS_DIR / f"{name}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = setup_logging()


# --- Retry with exponential backoff ---

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    retry_on: tuple = (requests.exceptions.RequestException,),
):
    """Decorator: retry z exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    if attempt == max_retries:
                        log.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    log.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retry in {delay:.1f}s")
                    time.sleep(delay)
        return wrapper
    return decorator


# --- Common data structures ---

@dataclass
class DetectionResult:
    chunk_id: int
    detector: str
    ai_probability: float      # 0-100, -1 = error
    human_probability: float   # 0-100
    classification: str        # "AI", "Human", "Mixed", "error"
    details: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.ai_probability >= 0

    @property
    def risk_label(self) -> str:
        return THRESHOLDS.label_pl(self.ai_probability)

    @property
    def risk_emoji(self) -> str:
        return THRESHOLDS.emoji(self.ai_probability)

    def to_dict(self) -> dict:
        return asdict(self)


# --- Text loading and cleaning ---

def clean_text(text: str) -> str:
    """Usun artefakty LaTeX z tekstu."""
    for noise in LATEX_NOISE_PATTERNS:
        text = text.replace(noise, "")
    return text.strip()


def load_chunks(chunks_dir: Optional[Path] = None, min_length: int = 50) -> Dict[int, str]:
    """Zaladuj wszystkie chunki tekstu. Zwraca {chunk_id: text}."""
    cdir = chunks_dir or CHUNKS_DIR
    chunks = {}

    for f in sorted(cdir.glob("chunk_*.txt")):
        try:
            cid = int(f.stem.split("_")[1])
            text = f.read_text(encoding="utf-8")
            text = clean_text(text)
            if len(text) >= min_length:
                chunks[cid] = text
        except (ValueError, IndexError, OSError) as e:
            log.warning(f"Nie mozna zaladowac {f.name}: {e}")

    log.info(f"Zaladowano {len(chunks)} chunkow z {cdir}")
    return chunks


def load_chunks_as_list(chunks_dir: Optional[Path] = None) -> List[tuple]:
    """Zaladuj chunki jako lista (chunk_id, text). Kompatybilnosc wsteczna."""
    chunks = load_chunks(chunks_dir)
    return [(cid, text) for cid, text in sorted(chunks.items())]


# --- API helpers ---

DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


@retry_with_backoff(max_retries=3, base_delay=3.0)
def call_zerogpt(text: str, max_chars: int = 10000) -> float:
    """Test z ZeroGPT API. Zwraca AI% (0-100) lub -1 przy bledzie."""
    response = requests.post(
        API_ENDPOINTS["zerogpt"],
        headers={
            **DEFAULT_HEADERS,
            "Origin": "https://www.zerogpt.com",
            "Referer": "https://www.zerogpt.com/",
        },
        json={"input_text": text[:max_chars]},
        timeout=60,
    )

    if response.status_code == 429:
        raise requests.exceptions.RequestException("Rate limited (429)")

    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            return data.get("data", {}).get("fakePercentage", 0)

    log.warning(f"ZeroGPT unexpected response: {response.status_code}")
    return -1


@retry_with_backoff(max_retries=2, base_delay=2.0)
def call_contentdetector(text: str, max_chars: int = 5000) -> float:
    """Test z ContentDetector.AI. Zwraca AI% (0-100) lub -1."""
    response = requests.post(
        API_ENDPOINTS["contentdetector"],
        headers={
            **DEFAULT_HEADERS,
            "Origin": "https://contentdetector.ai",
        },
        json={"text": text[:max_chars]},
        timeout=60,
    )

    if response.status_code == 200:
        data = response.json()
        prob = data.get("probability", data.get("ai_probability", 0))
        return prob * 100 if prob <= 1 else prob

    return -1


# --- Result caching ---

def get_cache_path(detector: str, chunk_id: int) -> Path:
    cache_dir = RESULTS_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{detector}_{chunk_id}.json"


def get_cached_result(detector: str, chunk_id: int) -> Optional[DetectionResult]:
    """Zwroc cached result jesli istnieje (max 24h)."""
    cache_path = get_cache_path(detector, chunk_id)
    if not cache_path.exists():
        return None

    try:
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours > 24:
            return None

        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return DetectionResult(**data)
    except Exception:
        return None


def cache_result(result: DetectionResult) -> None:
    """Zapisz result do cache."""
    cache_path = get_cache_path(result.detector, result.chunk_id)
    cache_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --- Report helpers ---

def save_json_report(data: Any, filename: str) -> Path:
    path = RESULTS_DIR / filename
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    log.info(f"Zapisano: {path}")
    return path


def save_markdown_report(content: str, filename: str) -> Path:
    path = RESULTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    log.info(f"Zapisano: {path}")
    return path
