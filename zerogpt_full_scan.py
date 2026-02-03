"""
Extended ZeroGPT testing - test ALL chunks to find problematic ones.
ZeroGPT is updated for GPT-5, Claude 4, Gemini 3.
"""

import requests
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class ZeroGPTResult:
    chunk_id: int
    ai_probability: float
    is_human: bool
    is_gpt: bool
    feedback: List[str]
    text_preview: str


def test_zerogpt(text: str, chunk_id: int) -> ZeroGPTResult:
    """Test with ZeroGPT - supports GPT-5, Claude, Gemini."""
    
    url = "https://api.zerogpt.com/api/detect/detectText"
    
    try:
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Origin": "https://www.zerogpt.com",
                "Referer": "https://www.zerogpt.com/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            json={"input_text": text[:15000]},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("success"):
                result = data.get("data", {})
                
                return ZeroGPTResult(
                    chunk_id=chunk_id,
                    ai_probability=result.get("fakePercentage", 0),
                    is_human=result.get("isHuman", False),
                    is_gpt=result.get("isGpt", False),
                    feedback=result.get("feedback", [])[:3],
                    text_preview=text[:100]
                )
        
        return ZeroGPTResult(
            chunk_id=chunk_id,
            ai_probability=-1,
            is_human=False,
            is_gpt=False,
            feedback=[],
            text_preview=text[:100]
        )
        
    except Exception as e:
        return ZeroGPTResult(
            chunk_id=chunk_id,
            ai_probability=-1,
            is_human=False,
            is_gpt=False,
            feedback=[f"Error: {str(e)}"],
            text_preview=text[:100]
        )


def load_chunks(chunks_dir: Path) -> List[tuple]:
    """Load all chunks."""
    chunks = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.txt")):
        chunk_id = int(chunk_file.stem.split("_")[1])
        with open(chunk_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        for noise in ["[KOD ŹRÓDŁOWY POMINIĘTY]",
                      "[STRUKTURA KATALOGÓW]", "[TABELA]", "{}", "{ }", "{\\ }"]:
            text = text.replace(noise, "")
        text = text.strip()
        
        if len(text) > 100:
            chunks.append((chunk_id, text))
    
    return chunks


def main():
    print("=" * 60)
    print("🔍 ZEROGPT FULL SCAN")
    print("   Wykrywa: GPT-5, Claude 4.5, Gemini 3")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    chunks_dir = base_dir / "chunks"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    chunks = load_chunks(chunks_dir)
    print(f"\n📁 Testuję WSZYSTKIE {len(chunks)} chunków...")
    
    results = []
    
    for i, (chunk_id, text) in enumerate(chunks):
        result = test_zerogpt(text, chunk_id)
        results.append(result)
        
        if result.ai_probability >= 0:
            if result.ai_probability > 50:
                emoji = "🔴"
            elif result.ai_probability > 30:
                emoji = "🟡"
            else:
                emoji = "🟢"
            print(f"  {emoji} Chunk {chunk_id:2d}: {result.ai_probability:5.1f}% AI")
        else:
            print(f"  ❌ Chunk {chunk_id:2d}: błąd")
        
        # Rate limiting
        time.sleep(1.5)
    
    # Analysis
    print("\n" + "=" * 60)
    print("📊 ANALIZA WYNIKÓW")
    print("=" * 60)
    
    valid = [r for r in results if r.ai_probability >= 0]
    
    if valid:
        avg = sum(r.ai_probability for r in valid) / len(valid)
        max_result = max(valid, key=lambda x: x.ai_probability)
        min_result = min(valid, key=lambda x: x.ai_probability)
        
        flagged = [r for r in valid if r.ai_probability > 50]
        warning = [r for r in valid if 30 < r.ai_probability <= 50]
        safe = [r for r in valid if r.ai_probability <= 30]
        
        print(f"""
📈 STATYSTYKI:
   Przetestowano: {len(valid)} chunków
   Średnia AI: {avg:.1f}%
   Min: {min_result.ai_probability:.1f}% (Chunk {min_result.chunk_id})
   Max: {max_result.ai_probability:.1f}% (Chunk {max_result.chunk_id})

📋 KLASYFIKACJA:
   🟢 Bezpieczne (<30%): {len(safe)} chunków
   🟡 Ostrzeżenie (30-50%): {len(warning)} chunków
   🔴 Problem (>50%): {len(flagged)} chunków
""")
        
        # Overall verdict
        if avg < 20:
            verdict = "✅ BARDZO BEZPIECZNE - Tekst wygląda na w pełni ludzki"
        elif avg < 35:
            verdict = "🟢 BEZPIECZNE - Niskie ryzyko wykrycia"
        elif avg < 50:
            verdict = "⚠️ UMIARKOWANE RYZYKO - Rozważ poprawki w flagowanych fragmentach"
        elif avg < 70:
            verdict = "🔶 PODWYŻSZONE RYZYKO - Wymagane przepisanie niektórych fragmentów"
        else:
            verdict = "🚨 WYSOKIE RYZYKO - Konieczne znaczne przepisanie"
        
        print(f"🎯 WERDYKT: {verdict}")
        
        # List problematic chunks
        if flagged:
            print("\n⚠️ CHUNKI WYMAGAJĄCE POPRAWY (>50% AI):")
            for r in sorted(flagged, key=lambda x: -x.ai_probability):
                print(f"   🔴 Chunk {r.chunk_id}: {r.ai_probability:.1f}%")
                print(f"      Początek: \"{r.text_preview[:60]}...\"")
                if r.feedback:
                    print(f"      Flagowane: {r.feedback[0][:80]}...")
        
        if warning:
            print("\n⚠️ CHUNKI DO OBSERWACJI (30-50% AI):")
            for r in sorted(warning, key=lambda x: -x.ai_probability):
                print(f"   🟡 Chunk {r.chunk_id}: {r.ai_probability:.1f}%")
        
        # Recommendations
        print("\n" + "=" * 60)
        print("💡 REKOMENDACJE")
        print("=" * 60)
        
        if flagged:
            print(f"""
Chunki {', '.join(str(r.chunk_id) for r in flagged)} mają >50% AI.

Jak je poprawić:
1. PRZEPISZ własnymi słowami - nie kopiuj struktur zdań
2. DODAJ OSOBISTE KOMENTARZE:
   - "Podczas implementacji zauważyłem, że..."
   - "W trakcie testów okazało się, że..."
   - "Co ciekawe, model..."

3. ZMIEŃ STRUKTURĘ:
   - Mieszaj krótkie i długie zdania
   - Dodaj pytania retoryczne
   - Użyj przykładów z codziennego życia

4. DODAJ SZCZEGÓŁY:
   - Konkretne daty eksperymentów
   - Nazwy plików które edytowałeś
   - Czasy wykonania (np. "model uczył się 3h 22min")
""")
        else:
            print("""
✅ Twój tekst wygląda bezpiecznie!

ZeroGPT (jeden z najnowszych detektorów, zaktualizowany dla GPT-5, 
Claude 4.5 i Gemini 3) nie wykrył znaczących śladów AI.

Pamiętaj jednak:
- Detektory NIE SĄ 100% dokładne
- Różne uczelnie używają różnych narzędzi
- W razie wątpliwości - zachowaj historię edycji dokumentu
""")
    
    # Save results
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detector": "ZeroGPT",
        "chunks_tested": len(valid),
        "average_ai_probability": avg if valid else 0,
        "flagged_count": len(flagged) if valid else 0,
        "results": [asdict(r) for r in results]
    }
    
    report_path = results_dir / "zerogpt_full_scan.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Wyniki zapisane: {report_path}")


if __name__ == "__main__":
    main()
