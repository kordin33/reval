# ==============================================================================
# 🔍 AI DETECTION TEST - INSTRUKCJA UŻYCIA
# ==============================================================================
# 
# Twój tekst jest wyekstrahowany i gotowy do testów.
# Aby przetestować go na detektorach AI, wykonaj poniższe kroki:
#
# ==============================================================================
# KROK 1: Uzyskaj darmowe klucze API (5-10 minut)
# ==============================================================================
#
# 1. SAPLING.AI (darmowy tier - 300 słów/dzień)
#    - Wejdź na: https://sapling.ai/user/settings
#    - Zarejestruj się (można przez Google)
#    - Skopiuj API Key
#
# 2. GPTZERO (darmowy tier)
#    - Wejdź na: https://gptzero.me/
#    - Zarejestruj się
#    - Przejdź do: https://app.gptzero.me/app/api
#    - Skopiuj API Key
#
# 3. ZEROGPT (opcjonalnie)
#    - Wejdź na: https://zerogpt.com/api
#    - Sprawdź dostępność darmowego tier
#
# ==============================================================================
# KROK 2: Ustaw zmienne środowiskowe
# ==============================================================================
#
# W PowerShell:
#   $env:SAPLING_API_KEY = "twoj_klucz_sapling"
#   $env:GPTZERO_API_KEY = "twoj_klucz_gptzero"
#
# Lub edytuj plik .env w tym katalogu
#
# ==============================================================================
# KROK 3: Uruchom testy
# ==============================================================================
#
#   python run_tests.py
#
# ==============================================================================

"""
Simplified AI detection test script.
Works with Sapling API (free tier available).
"""

import requests
import json
import time
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class DetectionResult:
    chunk_id: int
    detector: str
    ai_score: float  # 0-1 scale (0=human, 1=AI)
    ai_percentage: float  # 0-100%
    classification: str
    sentence_scores: List[Dict] = None
    error: Optional[str] = None


def test_with_sapling(text: str, chunk_id: int, api_key: str) -> DetectionResult:
    """Test text with Sapling.ai AI detector."""
    
    try:
        response = requests.post(
            "https://api.sapling.ai/api/v1/aidetect",
            json={
                "key": api_key,
                "text": text,
                "sent_scores": True
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            score = data.get("score", 0)
            
            # Get problematic sentences
            sentence_scores = []
            for sent in data.get("sentence_scores", []):
                if sent.get("score", 0) > 0.5:  # Flag high AI probability sentences
                    sentence_scores.append({
                        "sentence": sent["sentence"][:100] + "..." if len(sent["sentence"]) > 100 else sent["sentence"],
                        "score": sent["score"]
                    })
            
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Sapling",
                ai_score=score,
                ai_percentage=score * 100,
                classification="AI" if score > 0.5 else "Human",
                sentence_scores=sentence_scores[:5]  # Top 5 flagged sentences
            )
        else:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Sapling",
                ai_score=-1,
                ai_percentage=-1,
                classification="error",
                error=f"HTTP {response.status_code}: {response.text[:100]}"
            )
            
    except Exception as e:
        return DetectionResult(
            chunk_id=chunk_id,
            detector="Sapling",
            ai_score=-1,
            ai_percentage=-1,
            classification="error",
            error=str(e)
        )


def test_with_gptzero(text: str, chunk_id: int, api_key: str) -> DetectionResult:
    """Test text with GPTZero AI detector."""
    
    try:
        response = requests.post(
            "https://api.gptzero.me/v2/predict/text",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json"
            },
            json={"document": text},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if "documents" in data and data["documents"]:
                doc = data["documents"][0]
                ai_prob = doc.get("completely_generated_prob", 0)
                
                return DetectionResult(
                    chunk_id=chunk_id,
                    detector="GPTZero",
                    ai_score=ai_prob,
                    ai_percentage=ai_prob * 100,
                    classification=doc.get("predicted_class", "unknown"),
                    sentence_scores=[]
                )
            else:
                return DetectionResult(
                    chunk_id=chunk_id,
                    detector="GPTZero",
                    ai_score=-1,
                    ai_percentage=-1,
                    classification="error",
                    error="Invalid response format"
                )
        else:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="GPTZero",
                ai_score=-1,
                ai_percentage=-1,
                classification="error",
                error=f"HTTP {response.status_code}"
            )
            
    except Exception as e:
        return DetectionResult(
            chunk_id=chunk_id,
            detector="GPTZero",
            ai_score=-1,
            ai_percentage=-1,
            classification="error",
            error=str(e)
        )


def load_chunks(chunks_dir: Path) -> List[tuple]:
    """Load text chunks."""
    chunks = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.txt")):
        chunk_id = int(chunk_file.stem.split("_")[1])
        with open(chunk_file, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks.append((chunk_id, text))
    return chunks


def generate_report(results: List[DetectionResult], output_path: Path) -> str:
    """Generate detection report."""
    
    lines = [
        "# 🔍 RAPORT DETEKCJI AI",
        f"**Data:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 📊 PODSUMOWANIE",
        ""
    ]
    
    # Group by detector
    by_detector = {}
    for r in results:
        if r.detector not in by_detector:
            by_detector[r.detector] = []
        by_detector[r.detector].append(r)
    
    # Calculate averages
    for detector, res_list in by_detector.items():
        valid = [r for r in res_list if r.ai_score >= 0]
        if valid:
            avg = sum(r.ai_percentage for r in valid) / len(valid)
            
            if avg < 30:
                emoji = "✅"
                verdict = "NISKIE RYZYKO"
            elif avg < 50:
                emoji = "⚠️"
                verdict = "UMIARKOWANE RYZYKO"
            elif avg < 70:
                emoji = "🔶"
                verdict = "PODWYŻSZONE RYZYKO"
            else:
                emoji = "🚨"
                verdict = "WYSOKIE RYZYKO"
            
            lines.append(f"### {detector}")
            lines.append(f"- **Średni wynik AI:** {avg:.1f}%")
            lines.append(f"- **Werdykt:** {emoji} {verdict}")
            lines.append(f"- **Testowanych chunków:** {len(valid)}")
            lines.append("")
    
    # Per-chunk details
    lines.append("## 📝 SZCZEGÓŁY PER CHUNK")
    lines.append("")
    lines.append("| Chunk | Detektor | AI % | Klasyfikacja |")
    lines.append("|-------|----------|------|--------------|")
    
    for r in sorted(results, key=lambda x: (x.chunk_id, x.detector)):
        if r.ai_score >= 0:
            emoji = "🤖" if r.ai_percentage > 50 else "👤"
            lines.append(f"| {r.chunk_id} | {r.detector} | {r.ai_percentage:.1f}% | {emoji} {r.classification} |")
        else:
            lines.append(f"| {r.chunk_id} | {r.detector} | - | ❌ {r.error[:30]}... |")
    
    # Flagged sentences
    lines.append("")
    lines.append("## ⚠️ ZDANIA OZNACZONE JAKO AI")
    lines.append("")
    lines.append("Poniższe zdania mają wysokie prawdopodobieństwo wykrycia jako AI:")
    lines.append("")
    
    for r in results:
        if r.sentence_scores:
            lines.append(f"### Chunk {r.chunk_id} ({r.detector})")
            for sent in r.sentence_scores:
                lines.append(f"- `{sent['sentence']}` (score: {sent['score']:.2f})")
            lines.append("")
    
    # Recommendations
    lines.append("")
    lines.append("## 💡 REKOMENDACJE")
    lines.append("")
    
    valid_results = [r for r in results if r.ai_score >= 0]
    if valid_results:
        avg_all = sum(r.ai_percentage for r in valid_results) / len(valid_results)
        
        if avg_all < 30:
            lines.append("✅ **Twój tekst wygląda na napisany przez człowieka!**")
            lines.append("")
            lines.append("Wynik jest niski - prawdopodobnie nie będziesz miał problemów z detektorami AI.")
        elif avg_all < 50:
            lines.append("⚠️ **Umiarkowane ryzyko detekcji**")
            lines.append("")
            lines.append("Rozważ przepisanie najbardziej 'podejrzanych' fragmentów:")
            lines.append("- Dodaj osobiste anegdoty i przykłady")
            lines.append("- Zmień strukturę zdań na bardziej różnorodną")
            lines.append("- Użyj kolokwializmów gdzie to możliwe")
        else:
            lines.append("🚨 **Wysokie ryzyko detekcji**")
            lines.append("")
            lines.append("Zalecane działania:")
            lines.append("1. **Przeredaguj flagowane fragmenty** - przepisz je własnymi słowami")
            lines.append("2. **Dodaj osobiste komentarze** - np. 'W trakcie moich badań...'")
            lines.append("3. **Zróżnicuj styl** - mieszaj formalne i nieformalne wyrażenia")
            lines.append("4. **Dodaj szczegóły** - konkretne daty, nazwy, liczby")
    
    report = "\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


def main():
    print("=" * 60)
    print("🔍 AI DETECTION TEST")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    chunks_dir = base_dir / "chunks"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load chunks
    chunks = load_chunks(chunks_dir)
    print(f"\n📁 Załadowano {len(chunks)} chunków tekstu")
    
    # Check for API keys
    sapling_key = os.getenv("SAPLING_API_KEY")
    gptzero_key = os.getenv("GPTZERO_API_KEY")
    
    print("\n📋 Status kluczy API:")
    print(f"  SAPLING_API_KEY: {'✅ Ustawiony' if sapling_key else '❌ Brak'}")
    print(f"  GPTZERO_API_KEY: {'✅ Ustawiony' if gptzero_key else '❌ Brak'}")
    
    if not sapling_key and not gptzero_key:
        print("\n" + "=" * 60)
        print("⚠️  BRAK KLUCZY API!")
        print("=" * 60)
        print("""
Aby przetestować tekst, musisz uzyskać klucze API:

1. SAPLING.AI (Polecane - darmowy tier):
   → https://sapling.ai/user/settings
   → Zarejestruj się i skopiuj API Key

2. Ustaw klucz w PowerShell:
   $env:SAPLING_API_KEY = "twoj_klucz"

3. Uruchom ponownie ten skrypt

Alternatywnie: Możesz ręcznie przetestować fragmenty tekstu na:
   → https://sapling.ai/ai-content-detector
   → https://gptzero.me
   → https://zerogpt.com
   → https://copyleaks.com/ai-content-detector
        """)
        
        # Save chunks info for manual testing
        print("\n📄 Chunki zapisane do ręcznego testowania w:")
        print(f"   {chunks_dir}")
        print(f"\n   Przykładowy chunk (pierwszy 500 znaków):")
        print("-" * 60)
        print(chunks[0][1][:500])
        print("-" * 60)
        return
    
    # Run tests
    results = []
    
    # Sample 5 chunks evenly
    sample_size = min(5, len(chunks))
    step = max(1, len(chunks) // sample_size)
    sample_chunks = [chunks[i * step] for i in range(sample_size)]
    
    print(f"\n🔬 Testuję {len(sample_chunks)} chunków...")
    
    for chunk_id, text in sample_chunks:
        print(f"\n  Chunk {chunk_id}:")
        
        if sapling_key:
            result = test_with_sapling(text, chunk_id, sapling_key)
            results.append(result)
            if result.error:
                print(f"    Sapling: ❌ {result.error[:50]}")
            else:
                emoji = "🤖" if result.ai_percentage > 50 else "👤"
                print(f"    Sapling: {emoji} {result.ai_percentage:.1f}% AI")
        
        if gptzero_key:
            result = test_with_gptzero(text, chunk_id, gptzero_key)
            results.append(result)
            if result.error:
                print(f"    GPTZero: ❌ {result.error[:50]}")
            else:
                emoji = "🤖" if result.ai_percentage > 50 else "👤"
                print(f"    GPTZero: {emoji} {result.ai_percentage:.1f}% AI")
        
        time.sleep(1)  # Rate limiting
    
    # Generate report
    report_path = results_dir / "raport_ai_detection.md"
    report = generate_report(results, report_path)
    
    # Save raw results
    raw_path = results_dir / "raw_results.json"
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("📄 RAPORTY WYGENEROWANE:")
    print("=" * 60)
    print(f"  → {report_path}")
    print(f"  → {raw_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
