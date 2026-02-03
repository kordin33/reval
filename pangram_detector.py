"""
Pangram EditLens AI Detector - ICLR 2026, Open Source.

Bazuje na EditLens: najnowszy open-source detektor AI (2026).
Modele:
  - pangram/editlens_roberta-large (355M, szybki, CPU-friendly)
  - pangram/editlens_Llama-3.2-3B (3B, dokladniejszy, GPU)

Wymagania:
  pip install transformers torch
  Zaloguj sie na HuggingFace: huggingface-cli login
  (modele wymagaja akceptacji licencji CC BY-NC-SA 4.0)

Uzycie:
  python pangram_detector.py
  python pangram_detector.py --model llama   # uzyj modelu Llama 3B
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

try:
    from config import RESULTS_DIR, THRESHOLDS
    from utils import load_chunks, DetectionResult, setup_logging, save_json_report
except ImportError:
    RESULTS_DIR = Path(__file__).parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

log = None
try:
    log = setup_logging("pangram")
except Exception:
    import logging
    log = logging.getLogger("pangram")


# --- Model configs ---

MODELS = {
    "roberta": {
        "checkpoint": "pangram/editlens_roberta-large",
        "base_model": "FacebookAI/roberta-large",
        "max_length": 512,
        "description": "RoBERTa-large (355M) - szybki, CPU-friendly",
    },
    "llama": {
        "checkpoint": "pangram/editlens_Llama-3.2-3B",
        "base_model": "meta-llama/Llama-3.2-3B",
        "max_length": 1024,
        "description": "Llama 3.2 3B - dokladniejszy, wymaga GPU",
    },
}


class PangramDetector:
    """Detektor AI oparty na Pangram EditLens (ICLR 2026)."""

    def __init__(self, model_key: str = "roberta"):
        self.model_key = model_key
        self.model_config = MODELS[model_key]
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Zaladuj model z HuggingFace."""
        try:
            import torch
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError(
                "Zainstaluj: pip install transformers torch\n"
                "Zaloguj sie: huggingface-cli login"
            )

        checkpoint = self.model_config["checkpoint"]
        log.info(f"Ladowanie modelu: {checkpoint}")

        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if device == 0 else torch.float32

        try:
            self.pipeline = hf_pipeline(
                "text-classification",
                model=checkpoint,
                device=device,
                torch_dtype=dtype,
                truncation=True,
                max_length=self.model_config["max_length"],
            )
            log.info(f"Model zaladowany na {'GPU' if device == 0 else 'CPU'}")
        except Exception as e:
            # Fallback: try as a generic classifier
            log.warning(f"Pipeline error: {e}. Trying alternative loading...")
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint, torch_dtype=dtype
                )
                if device == 0:
                    self.model = self.model.cuda()
                self.model.eval()
                self._use_manual = True
                log.info("Model zaladowany (manual mode)")
            except Exception as e2:
                raise RuntimeError(
                    f"Nie mozna zaladowac modelu {checkpoint}: {e2}\n"
                    f"Upewnij sie, ze zaakceptowales licencje na HuggingFace:\n"
                    f"  https://huggingface.co/{checkpoint}\n"
                    f"I zalogowales sie: huggingface-cli login"
                )

    def detect(self, text: str, chunk_id: int = 0) -> DetectionResult:
        """Wykryj tekst AI."""
        if not text or len(text.strip()) < 30:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Pangram-EditLens",
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                error="Tekst za krotki (min. 30 znakow)",
            )

        try:
            if self.pipeline:
                result = self.pipeline(text[:4000])
                if isinstance(result, list):
                    result = result[0]

                label = result.get("label", "").upper()
                score = result.get("score", 0.0)

                # EditLens score: 0=human, 1=AI
                if "AI" in label or "FAKE" in label or "LABEL_1" in label:
                    ai_prob = score * 100
                elif "HUMAN" in label or "REAL" in label or "LABEL_0" in label:
                    ai_prob = (1 - score) * 100
                else:
                    # Use score_pred if available
                    ai_prob = score * 100

                return DetectionResult(
                    chunk_id=chunk_id,
                    detector="Pangram-EditLens",
                    ai_probability=ai_prob,
                    human_probability=100 - ai_prob,
                    classification="AI" if ai_prob > 50 else "Human",
                    details={"model": self.model_key, "raw_label": label, "raw_score": score},
                )
            else:
                return DetectionResult(
                    chunk_id=chunk_id,
                    detector="Pangram-EditLens",
                    ai_probability=-1,
                    human_probability=-1,
                    classification="error",
                    error="Model nie zaladowany",
                )

        except Exception as e:
            log.error(f"Pangram detect error chunk {chunk_id}: {e}")
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Pangram-EditLens",
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                error=str(e)[:100],
            )

    def detect_batch(self, chunks: Dict[int, str]) -> List[DetectionResult]:
        """Wykryj AI w wielu chunkach."""
        results = []
        total = len(chunks)

        for i, (cid, text) in enumerate(sorted(chunks.items())):
            log.info(f"Pangram: chunk {cid} ({i+1}/{total})")
            result = self.detect(text, chunk_id=cid)
            results.append(result)

            if result.is_valid:
                emoji = THRESHOLDS.emoji(result.ai_probability)
                print(f"  Chunk {cid:2d}: {emoji} {result.ai_probability:.1f}% AI")
            else:
                print(f"  Chunk {cid:2d}: ERR {result.error}")

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pangram EditLens AI Detector")
    parser.add_argument(
        "--model", choices=["roberta", "llama"], default="roberta",
        help="Model do uzycia (roberta=szybki, llama=dokladniejszy)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"PANGRAM EDITLENS AI DETECTOR (ICLR 2026)")
    print(f"Model: {MODELS[args.model]['description']}")
    print("=" * 60)

    detector = PangramDetector(model_key=args.model)

    chunks = load_chunks()
    if not chunks:
        print("Brak chunkow. Uzyj: python main.py --input praca.tex")
        return

    print(f"\nAnalizuje {len(chunks)} chunkow...\n")

    results = detector.detect_batch(chunks)

    # Summary
    valid = [r for r in results if r.is_valid]
    if valid:
        avg = sum(r.ai_probability for r in valid) / len(valid)
        print(f"\nSrednia: {THRESHOLDS.emoji(avg)} {avg:.1f}% AI")
        print(f"Werdykt: {THRESHOLDS.label_pl(avg)}")

        flagged = [r for r in valid if r.ai_probability > 50]
        if flagged:
            print(f"Flagowane chunki: {[r.chunk_id for r in flagged]}")

    # Save
    save_json_report(
        [r.to_dict() for r in results],
        "pangram_results.json",
    )
    print(f"\nWyniki zapisane w: {RESULTS_DIR / 'pangram_results.json'}")


if __name__ == "__main__":
    main()
