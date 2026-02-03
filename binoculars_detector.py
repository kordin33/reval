"""
Binoculars: Zero-Shot Detection of Machine-Generated Text.

Implementacja metody z artykulu:
  "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text" (2024)

Idea: oblicz stosunek perplexity tekstu pod dwoma roznymi modelami jezykowymi.
  - Tekst ludzki -> rozne modele daja rozne prawdopodobienstwa -> wysoki stosunek
  - Tekst AI -> wszystkie modele sie zgadzaja -> niski stosunek (bliski 1.0)

Score = ppl(observer) / ppl(performer)
Jezeli score < threshold -> tekst wygenerowany przez AI.

Modele dobrane do 8GB VRAM (RTX 4060 Ti):
  - Observer:  meta-llama/Llama-3.2-1B  (~2GB fp16)
  - Performer: meta-llama/Llama-3.2-3B  (~6GB fp16)
  - Fallback:  gpt2 / gpt2-xl            (~3GB lacznie)
"""

import os
import sys
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Brak wymaganych bibliotek. Instaluje: torch, transformers")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "torch", "transformers", "accelerate", "-q"])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BINOCULARS_CONFIG, THRESHOLDS, RESULTS_DIR
from utils import DetectionResult, setup_logging, load_chunks, save_json_report


log = setup_logging("binoculars")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_device() -> torch.device:
    """Auto-detect GPU, fallback to CPU."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_mem / 1024**2
        log.info(f"GPU wykryty: {gpu_name} ({vram_mb:.0f} MB VRAM)")
        return torch.device("cuda")
    log.warning("Brak GPU - uzywam CPU (bedzie wolniej)")
    return torch.device("cpu")


def _score_to_ai_probability(score: float, threshold: float) -> float:
    """
    Konwertuj binoculars score na AI probability (0-100%).

    Uzywa sigmoid-like mapping wycentrowanego na thresholdzie:
      - score << threshold  ->  ~100% AI
      - score == threshold  ->  ~50%
      - score >> threshold  ->  ~0% AI

    Parametr k kontroluje stromizne krzywej.
    """
    k = 25.0  # steepness
    exponent = -k * (score - threshold)
    # Clamp exponent to avoid overflow
    exponent = max(min(exponent, 500.0), -500.0)
    try:
        prob = 1.0 / (1.0 + math.exp(exponent))
    except OverflowError:
        prob = 0.0 if exponent > 0 else 1.0
    return round(prob * 100, 2)


# ---------------------------------------------------------------------------
# BinocularsDetector
# ---------------------------------------------------------------------------

class BinocularsDetector:
    """
    Detektor AI oparty na metodzie Binoculars.

    Uzywa dwoch modeli jezykowych (observer i performer) do obliczenia
    stosunku perplexity. Niski stosunek oznacza tekst AI.
    """

    DETECTOR_NAME = "Binoculars"

    def __init__(
        self,
        observer_model: Optional[str] = None,
        performer_model: Optional[str] = None,
        device: Optional[str] = None,
        threshold_mode: str = "low_fpr",
    ):
        """
        Args:
            observer_model:  nazwa/sciezka mniejszego modelu (domyslnie z config)
            performer_model: nazwa/sciezka wiekszego modelu (domyslnie z config)
            device:          "cuda", "cpu" lub "auto"
            threshold_mode:  "low_fpr" (~0.01% FPR) lub "balanced"
        """
        cfg = BINOCULARS_CONFIG

        self.observer_name = observer_model or cfg.observer_model
        self.performer_name = performer_model or cfg.performer_model
        self.max_length = cfg.max_length
        self.threshold_mode = threshold_mode

        if threshold_mode == "balanced":
            self.threshold = cfg.threshold_balanced
        else:
            self.threshold = cfg.threshold_low_fpr

        # Device
        device_str = device or cfg.device
        if device_str == "auto":
            self.device = _detect_device()
        else:
            self.device = torch.device(device_str)

        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Load models
        self.observer_model, self.observer_tokenizer = self._load_model_pair(
            self.observer_name, role="observer"
        )
        self.performer_model, self.performer_tokenizer = self._load_model_pair(
            self.performer_name, role="performer"
        )

        log.info(
            f"Binoculars gotowy: observer={self.observer_name}, "
            f"performer={self.performer_name}, device={self.device}, "
            f"threshold={self.threshold} ({self.threshold_mode})"
        )

    def _load_model_pair(
        self, model_name: str, role: str
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Zaladuj model i tokenizer. Probuj Llama, fallback do GPT-2.
        """
        fallback_map = {
            "meta-llama/Llama-3.2-1B": "gpt2",
            "meta-llama/Llama-3.2-3B": "gpt2-xl",
        }

        try:
            log.info(f"Ladowanie {role}: {model_name}...")
            model, tokenizer = self._load_single(model_name)
            log.info(f"  {role} zaladowany: {model_name}")
            return model, tokenizer
        except Exception as e:
            log.warning(f"  Nie mozna zaladowac {model_name}: {e}")

            fallback = fallback_map.get(model_name)
            if fallback:
                log.info(f"  Probuje fallback: {fallback}...")
                try:
                    model, tokenizer = self._load_single(fallback)
                    # Update stored name so logs are accurate
                    if role == "observer":
                        self.observer_name = fallback
                    else:
                        self.performer_name = fallback
                    log.info(f"  {role} fallback zaladowany: {fallback}")
                    return model, tokenizer
                except Exception as e2:
                    raise RuntimeError(
                        f"Nie mozna zaladowac ani {model_name} ani {fallback}: {e2}"
                    ) from e2
            else:
                raise

    def _load_single(
        self, model_name: str
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Zaladuj pojedynczy model na wlasciwe urzadzenie."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map=self.device.type if self.device.type == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except torch.cuda.OutOfMemoryError:
            log.error(
                f"OOM przy ladowaniu {model_name}! "
                f"Sprobuj mniejsze modele (gpt2/gpt2-xl) lub ustaw device='cpu'."
            )
            raise
        except Exception:
            raise

        if self.device.type == "cpu":
            model = model.to(self.device)

        model.eval()
        return model, tokenizer

    @torch.no_grad()
    def _compute_perplexity(
        self,
        text: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ) -> float:
        """
        Oblicz perplexity tekstu pod danym modelem.

        Perplexity = exp( -1/N * sum(log P(token_i | context)) )
        """
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        input_ids = encodings.input_ids.to(model.device)

        if input_ids.size(1) < 2:
            log.warning("Tekst za krotki do obliczenia perplexity")
            return float("inf")

        # Labels shifted inside the model, but we need to mask padding
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        # outputs.loss is the mean cross-entropy over all tokens
        neg_log_likelihood = outputs.loss

        perplexity = torch.exp(neg_log_likelihood).item()
        return perplexity

    def _compute_cross_perplexity(self, text: str) -> Tuple[float, float, float]:
        """
        Oblicz binoculars score = ppl(observer) / ppl(performer).

        Returns:
            (score, observer_ppl, performer_ppl)
        """
        observer_ppl = self._compute_perplexity(
            text, self.observer_model, self.observer_tokenizer
        )
        performer_ppl = self._compute_perplexity(
            text, self.performer_model, self.performer_tokenizer
        )

        if performer_ppl == 0 or math.isinf(performer_ppl) or math.isnan(performer_ppl):
            log.warning("Performer perplexity nieprawidlowe, zwracam inf")
            return float("inf"), observer_ppl, performer_ppl

        if math.isinf(observer_ppl) or math.isnan(observer_ppl):
            log.warning("Observer perplexity nieprawidlowe, zwracam inf")
            return float("inf"), observer_ppl, performer_ppl

        score = observer_ppl / performer_ppl
        return score, observer_ppl, performer_ppl

    def detect(self, text: str, chunk_id: int = 0) -> DetectionResult:
        """
        Glowna metoda detekcji. Zwraca DetectionResult.

        Args:
            text:     tekst do analizy
            chunk_id: identyfikator fragmentu (chunka)
        """
        if not text or len(text.strip()) < 20:
            return DetectionResult(
                chunk_id=chunk_id,
                detector=self.DETECTOR_NAME,
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                error="Tekst za krotki (min. 20 znakow)",
            )

        try:
            score, obs_ppl, perf_ppl = self._compute_cross_perplexity(text)

            if math.isinf(score) or math.isnan(score):
                return DetectionResult(
                    chunk_id=chunk_id,
                    detector=self.DETECTOR_NAME,
                    ai_probability=-1,
                    human_probability=-1,
                    classification="error",
                    error="Nie mozna obliczyc score (perplexity error)",
                )

            ai_prob = _score_to_ai_probability(score, self.threshold)
            human_prob = round(100.0 - ai_prob, 2)

            if ai_prob >= 50:
                classification = "AI"
            elif ai_prob >= 30:
                classification = "Mixed"
            else:
                classification = "Human"

            return DetectionResult(
                chunk_id=chunk_id,
                detector=self.DETECTOR_NAME,
                ai_probability=ai_prob,
                human_probability=human_prob,
                classification=classification,
                details={
                    "binoculars_score": round(score, 6),
                    "observer_perplexity": round(obs_ppl, 4),
                    "performer_perplexity": round(perf_ppl, 4),
                    "threshold": self.threshold,
                    "threshold_mode": self.threshold_mode,
                    "observer_model": self.observer_name,
                    "performer_model": self.performer_name,
                    "text_length": len(text),
                },
            )

        except torch.cuda.OutOfMemoryError:
            log.error(
                f"OOM na chunk {chunk_id}! Tekst za dlugi lub za malo VRAM. "
                "Sprobuj krotsze chunki, mniejsze modele, lub device='cpu'."
            )
            return DetectionResult(
                chunk_id=chunk_id,
                detector=self.DETECTOR_NAME,
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                error="CUDA Out of Memory - za malo VRAM",
            )
        except Exception as e:
            log.error(f"Blad detekcji chunk {chunk_id}: {e}")
            return DetectionResult(
                chunk_id=chunk_id,
                detector=self.DETECTOR_NAME,
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                error=str(e)[:200],
            )

    def detect_batch(self, texts: Dict[int, str]) -> List[DetectionResult]:
        """
        Batch detekcja na wielu chunkach.

        Args:
            texts: dict {chunk_id: text}

        Returns:
            Lista DetectionResult
        """
        results = []
        total = len(texts)

        for i, (chunk_id, text) in enumerate(sorted(texts.items()), 1):
            log.info(f"  [{i}/{total}] Chunk {chunk_id} ({len(text)} znakow)...")
            t0 = time.time()
            result = self.detect(text, chunk_id)
            elapsed = time.time() - t0

            if result.is_valid:
                score = result.details.get("binoculars_score", "?")
                log.info(
                    f"    -> score={score}, AI={result.ai_probability:.1f}%, "
                    f"klasa={result.classification} ({elapsed:.1f}s)"
                )
            else:
                log.warning(f"    -> BLAD: {result.error} ({elapsed:.1f}s)")

            results.append(result)

            # Clear GPU cache between chunks
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return results


# ---------------------------------------------------------------------------
# Standalone main
# ---------------------------------------------------------------------------

def print_results_table(results: List[DetectionResult]) -> None:
    """Wyswietl tabelke wynikow w konsoli."""
    print()
    print("=" * 90)
    print(f"{'Chunk':>6} | {'Score':>8} | {'AI %':>7} | {'Klasa':>8} | "
          f"{'Obs PPL':>10} | {'Perf PPL':>10} | {'Ryzyko'}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: x.chunk_id):
        if r.is_valid:
            d = r.details
            score = d.get("binoculars_score", 0)
            obs = d.get("observer_perplexity", 0)
            perf = d.get("performer_perplexity", 0)
            print(
                f"{r.chunk_id:>6} | {score:>8.4f} | {r.ai_probability:>6.1f}% | "
                f"{r.classification:>8} | {obs:>10.2f} | {perf:>10.2f} | "
                f"{r.risk_emoji} {r.risk_label}"
            )
        else:
            print(f"{r.chunk_id:>6} | {'ERR':>8} | {'---':>7} | {'error':>8} | "
                  f"{'---':>10} | {'---':>10} | {r.error}")

    print("=" * 90)

    # Summary
    valid = [r for r in results if r.is_valid]
    if valid:
        avg_ai = sum(r.ai_probability for r in valid) / len(valid)
        avg_score = sum(
            r.details.get("binoculars_score", 0) for r in valid
        ) / len(valid)
        print(f"\nSrednia: score={avg_score:.4f}, AI={avg_ai:.1f}%")
        print(f"Klasyfikacja ogolna: {THRESHOLDS.emoji(avg_ai)} {THRESHOLDS.label_pl(avg_ai)}")


def main():
    """Uruchom Binoculars na wszystkich chunkach z katalogu chunks/."""
    print("=" * 60)
    print("  BINOCULARS - Zero-Shot AI Text Detection")
    print("=" * 60)
    print()

    # Load chunks
    chunks = load_chunks()
    if not chunks:
        print("Brak chunkow do analizy w katalogu chunks/")
        sys.exit(1)

    print(f"Zaladowano {len(chunks)} chunkow\n")

    # Initialize detector
    try:
        detector = BinocularsDetector()
    except RuntimeError as e:
        print(f"\nBlad inicjalizacji detektora: {e}")
        print("\nMozliwe rozwiazania:")
        print("  1. Zaloguj sie do HuggingFace: huggingface-cli login")
        print("  2. Ustaw token: export HF_TOKEN=hf_...")
        print("  3. Uzyj mniejszych modeli (gpt2/gpt2-xl) - edytuj config.py")
        sys.exit(1)

    # Run detection
    print(f"\nUruchamiam detekcje na {len(chunks)} chunkach...\n")
    t_start = time.time()

    results = detector.detect_batch(chunks)

    elapsed_total = time.time() - t_start
    print(f"\nCzas calkowity: {elapsed_total:.1f}s "
          f"(srednio {elapsed_total / len(chunks):.1f}s/chunk)")

    # Print table
    print_results_table(results)

    # Save JSON results
    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detector": "Binoculars",
        "observer_model": detector.observer_name,
        "performer_model": detector.performer_name,
        "threshold": detector.threshold,
        "threshold_mode": detector.threshold_mode,
        "device": str(detector.device),
        "total_chunks": len(chunks),
        "total_time_s": round(elapsed_total, 1),
        "results": [r.to_dict() for r in results],
    }

    out_path = save_json_report(results_data, "binoculars_results.json")
    print(f"\nWyniki zapisane: {out_path}")


if __name__ == "__main__":
    main()
