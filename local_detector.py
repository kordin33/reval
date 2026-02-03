"""
AI Detection using HuggingFace models - NO API KEY REQUIRED!
Uses local models for AI text detection.

Models used:
1. roberta-base-openai-detector - OpenAI's GPT-2 output detector (legacy)
2. Hello-SimpleAI/chatgpt-detector-roberta - ChatGPT detector
3. andreas122001/roberta-mixed-detector - Mixed model detector
4. TrustSafeAI/RADAR-Vicuna-7B - RADAR adversarial detector (optional, large)
"""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from config import RESULTS_DIR
    from utils import load_chunks_as_list, setup_logging, DetectionResult as SharedResult
except ImportError:
    RESULTS_DIR = Path(__file__).parent / "results"


def load_detectors(include_large: bool = False):
    """Load AI detection models from HuggingFace."""
    if not HAS_TRANSFORMERS:
        print("Brak transformers/torch. Instaluj: pip install transformers torch")
        return {}

    detectors = {}
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"

    print(f"\nLadowanie modeli na {device_name}...")

    models = [
        ("OpenAI-RoBERTa", "openai-community/roberta-base-openai-detector"),
        ("ChatGPT-Detector", "Hello-SimpleAI/chatgpt-detector-roberta"),
        ("RoBERTa-Mixed", "andreas122001/roberta-mixed-detector"),
    ]

    for name, model_id in models:
        try:
            print(f"  -> {model_id}...")
            detectors[name] = pipeline(
                "text-classification",
                model=model_id,
                device=device,
            )
            print(f"  OK {name}")
        except Exception as e:
            print(f"  SKIP {name}: {str(e)[:60]}")

    return detectors


def detect_ai(text: str, detector, detector_name: str, chunk_id: int) -> DetectionResult:
    """Run AI detection on text."""
    
    # Truncate text if too long (model limit ~512 tokens)
    max_chars = 1500
    truncated = text[:max_chars] if len(text) > max_chars else text
    
    try:
        result = detector(truncated)
        
        # Parse results (format varies by model)
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        label = result.get("label", "").upper()
        score = result.get("score", 0)
        
        # Normalize to AI probability
        # Different models use different labels
        if "FAKE" in label or "AI" in label or "GPT" in label or "MACHINE" in label:
            ai_prob = score
            human_prob = 1 - score
            classification = "AI" if score > 0.5 else "Human"
        elif "REAL" in label or "HUMAN" in label:
            human_prob = score
            ai_prob = 1 - score
            classification = "Human" if score > 0.5 else "AI"
        else:
            # Unknown label format - use score directly
            ai_prob = score if score > 0.5 else 1 - score
            human_prob = 1 - ai_prob
            classification = label
        
        return DetectionResult(
            chunk_id=chunk_id,
            detector_name=detector_name,
            ai_probability=ai_prob * 100,
            human_probability=human_prob * 100,
            classification=classification,
            raw_scores={"label": label, "score": score},
            text_preview=truncated[:100] + "..."
        )
        
    except Exception as e:
        return DetectionResult(
            chunk_id=chunk_id,
            detector_name=detector_name,
            ai_probability=-1,
            human_probability=-1,
            classification=f"ERROR: {str(e)[:50]}",
            raw_scores={},
            text_preview=truncated[:100] + "..."
        )


def load_chunks(chunks_dir: Path) -> List[tuple]:
    """Load text chunks."""
    chunks = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.txt")):
        chunk_id = int(chunk_file.stem.split("_")[1])
        with open(chunk_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean text
        text = text.replace("[KOD ŹRÓDŁOWY POMINIĘTY]", "")
        text = text.replace("[STRUKTURA KATALOGÓW]", "")
        text = text.replace("[TABELA]", "")
        text = text.replace("{}", "").replace("{ }", "").replace("{\\ }", "")
        text = text.strip()
        
        chunks.append((chunk_id, text))
    return chunks


def generate_report(results: List[DetectionResult], output_path: Path) -> str:
    """Generate markdown report."""
    
    lines = [
        "# 🔍 RAPORT DETEKCJI AI (Lokalne modele HuggingFace)",
        "",
        f"**Data:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## ℹ️ Użyte modele",
        "",
        "1. **OpenAI-RoBERTa** - `openai-community/roberta-base-openai-detector`",
        "2. **ChatGPT-Detector** - `Hello-SimpleAI/chatgpt-detector-roberta`",
        "3. **RoBERTa-Mixed** - `andreas122001/roberta-mixed-detector`",
        "",
        "---",
        "",
    ]
    
    # Calculate averages by detector
    by_detector = {}
    for r in results:
        if r.detector_name not in by_detector:
            by_detector[r.detector_name] = []
        if r.ai_probability >= 0:
            by_detector[r.detector_name].append(r)
    
    # Overall summary
    lines.append("## 📊 PODSUMOWANIE")
    lines.append("")
    
    all_ai_probs = [r.ai_probability for r in results if r.ai_probability >= 0]
    if all_ai_probs:
        overall_avg = sum(all_ai_probs) / len(all_ai_probs)
        
        if overall_avg < 30:
            verdict = "✅ NISKIE RYZYKO - Tekst wygląda na ludzki"
            color = "green"
        elif overall_avg < 50:
            verdict = "⚠️ UMIARKOWANE RYZYKO - Możliwe fałszywe alarmy"
            color = "yellow"
        elif overall_avg < 70:
            verdict = "🔶 PODWYŻSZONE RYZYKO - Rozważ edycję niektórych fragmentów"
            color = "orange"
        else:
            verdict = "🚨 WYSOKIE RYZYKO - Wymaga przepisania fragmentów"
            color = "red"
        
        lines.append(f"### {verdict}")
        lines.append("")
        lines.append(f"**Średnia wszystkich detektorów:** {overall_avg:.1f}% AI")
        lines.append("")
    
    # Per-detector summary
    lines.append("### Wyniki per detektor:")
    lines.append("")
    lines.append("| Detektor | Średnia AI % | Chunków | Werdykt |")
    lines.append("|----------|--------------|---------|---------|")
    
    for detector_name, detector_results in by_detector.items():
        if detector_results:
            avg = sum(r.ai_probability for r in detector_results) / len(detector_results)
            emoji = "🤖" if avg > 50 else "👤"
            verdict = "AI" if avg > 50 else "Human"
            lines.append(f"| {detector_name} | {avg:.1f}% | {len(detector_results)} | {emoji} {verdict} |")
    
    lines.append("")
    
    # Detailed results per chunk
    lines.append("## 📝 WYNIKI PER CHUNK")
    lines.append("")
    lines.append("| Chunk | OpenAI-RoBERTa | ChatGPT-Det | RoBERTa-Mixed | Średnia |")
    lines.append("|-------|----------------|-------------|---------------|---------|")
    
    # Group by chunk
    by_chunk = {}
    for r in results:
        if r.chunk_id not in by_chunk:
            by_chunk[r.chunk_id] = {}
        by_chunk[r.chunk_id][r.detector_name] = r
    
    for chunk_id in sorted(by_chunk.keys()):
        chunk_results = by_chunk[chunk_id]
        
        vals = []
        row = [f"{chunk_id}"]
        
        for det_name in ["OpenAI-RoBERTa", "ChatGPT-Detector", "RoBERTa-Mixed"]:
            if det_name in chunk_results:
                r = chunk_results[det_name]
                if r.ai_probability >= 0:
                    vals.append(r.ai_probability)
                    emoji = "🤖" if r.ai_probability > 50 else "👤"
                    row.append(f"{emoji} {r.ai_probability:.0f}%")
                else:
                    row.append("❌")
            else:
                row.append("-")
        
        if vals:
            avg = sum(vals) / len(vals)
            row.append(f"{avg:.0f}%")
        else:
            row.append("-")
        
        lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")
    
    # Flagged chunks
    flagged = []
    for chunk_id, chunk_results in by_chunk.items():
        vals = [r.ai_probability for r in chunk_results.values() if r.ai_probability >= 0]
        if vals:
            avg = sum(vals) / len(vals)
            if avg > 60:
                flagged.append((chunk_id, avg))
    
    if flagged:
        lines.append("## ⚠️ CHUNKI WYMAGAJĄCE UWAGI (>60% AI)")
        lines.append("")
        for chunk_id, avg in sorted(flagged, key=lambda x: -x[1]):
            lines.append(f"- **Chunk {chunk_id}**: {avg:.0f}% - rozważ przepisanie")
        lines.append("")
    
    # Recommendations
    lines.append("## 💡 REKOMENDACJE")
    lines.append("")
    
    if all_ai_probs and overall_avg < 40:
        lines.append("✅ **Twój tekst jest prawdopodobnie bezpieczny!**")
        lines.append("")
        lines.append("Lokalne modele AI nie wykryły znaczących śladów sztucznej inteligencji.")
        lines.append("Pamiętaj jednak, że detektory AI są zawodne i mogą dawać fałszywe wyniki.")
    else:
        lines.append("### Jak zmniejszyć ryzyko wykrycia:")
        lines.append("")
        lines.append("1. **Przepisz flagowane fragmenty** własnymi słowami")
        lines.append("2. **Dodaj osobiste komentarze** - 'W trakcie moich badań...'")
        lines.append("3. **Zróżnicuj strukturę zdań** - mieszaj krótkie i długie")
        lines.append("4. **Użyj kolokwializmów** gdzie to pasuje")
        lines.append("5. **Dodaj konkretne szczegóły** - daty, nazwy, liczby")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Raport wygenerowany automatycznie przy użyciu modeli HuggingFace*")
    
    report = "\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


def main():
    print("=" * 60)
    print("🔍 AI DETECTION - LOKALNE MODELE (BEZ API)")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    chunks_dir = base_dir / "chunks"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load chunks
    chunks = load_chunks(chunks_dir)
    print(f"\n📁 Załadowano {len(chunks)} chunków tekstu")
    
    # Load detectors
    detectors = load_detectors()
    
    if not detectors:
        print("\n❌ Nie udało się załadować żadnego modelu!")
        return
    
    print(f"\n✅ Załadowano {len(detectors)} detektorów")
    
    # Sample chunks for testing (every 4th chunk for speed)
    sample_ids = [1, 4, 7, 10, 13, 16, 19, 22, 24]
    sample_chunks = [(cid, text) for cid, text in chunks if cid in sample_ids]
    
    if not sample_chunks:
        sample_chunks = chunks[:9]
    
    print(f"\n🔬 Testuję {len(sample_chunks)} chunków na {len(detectors)} detektorach...")
    print("   (to może potrwać kilka minut)")
    
    results = []
    
    for chunk_id, text in sample_chunks:
        print(f"\n  📄 Chunk {chunk_id}:")
        
        for detector_name, detector in detectors.items():
            result = detect_ai(text, detector, detector_name, chunk_id)
            results.append(result)
            
            if result.ai_probability >= 0:
                emoji = "🤖" if result.ai_probability > 50 else "👤"
                print(f"     {detector_name}: {emoji} {result.ai_probability:.0f}% AI")
            else:
                print(f"     {detector_name}: ❌ {result.classification[:30]}")
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 GENEROWANIE RAPORTU...")
    print("=" * 60)
    
    report_path = results_dir / "RAPORT_AI_DETECTION.md"
    report = generate_report(results, report_path)
    
    # Save raw results
    raw_path = results_dir / "raw_huggingface_results.json"
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Raporty zapisane:")
    print(f"   → {report_path}")
    print(f"   → {raw_path}")
    
    print("\n" + "=" * 60)
    print("📋 WYNIKI:")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
