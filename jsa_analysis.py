"""
JSA-LIKE TESTING FRAMEWORK
Uses perplexity-based detection similar to Polish university systems.

This script:
1. Calculates perplexity for each chunk (similar to JSA methodology)
2. Tests on multiple detectors including Polish ones
3. Shows EXACTLY which chunks and sentences are problematic
4. Generates detailed report with specific recommendations

Since JSA is not publicly available, we simulate its methodology.
"""

import json
import time
import math
import requests
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple
from collections import Counter
import re


@dataclass
class ChunkAnalysis:
    chunk_id: int
    text_preview: str
    char_count: int
    word_count: int
    avg_sentence_length: float
    sentence_length_variance: float
    unique_word_ratio: float
    estimated_perplexity: str  # "low", "medium", "high"
    zerogpt_score: float
    risk_level: str
    flagged_sentences: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def calculate_sentence_stats(text: str) -> Tuple[float, float, List[str]]:
    """Calculate average sentence length and variance."""
    # Split by sentence endings
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return 0, 0, []
    
    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths) / len(lengths)
    variance = sum((l - avg) ** 2 for l in lengths) / len(lengths) if len(lengths) > 1 else 0
    
    return avg, variance, sentences


def estimate_perplexity_risk(text: str) -> Tuple[str, List[str]]:
    """
    Estimate perplexity risk based on text characteristics.
    Low variance = low perplexity = flagged as AI by JSA.
    
    Returns: (risk_level, flagged_patterns)
    """
    flagged = []
    risk_score = 0
    
    # Check for formal patterns (low perplexity indicators)
    formal_patterns = [
        (r"z punktu widzenia", "Formalna fraza: 'z punktu widzenia'"),
        (r"należy podkreślić", "Formalna fraza: 'należy podkreślić'"),
        (r"w niniejszej pracy", "Formalna fraza: 'w niniejszej pracy'"),
        (r"podsumowując,?\s*można", "Formalna fraza: 'podsumowując można'"),
        (r"w związku z powyższym", "Formalna fraza: 'w związku z powyższym'"),
        (r"mając na uwadze", "Formalna fraza: 'mając na uwadze'"),
        (r"istotnym aspektem jest", "Formalna fraza: 'istotnym aspektem jest'"),
        (r"zagadnienie to", "Formalna fraza: 'zagadnienie to'"),
        (r"powyższe rozważania", "Formalna fraza: 'powyższe rozważania'"),
        (r"w kontekście", "Formalna fraza: 'w kontekście'"),
    ]
    
    text_lower = text.lower()
    for pattern, desc in formal_patterns:
        if re.search(pattern, text_lower):
            flagged.append(desc)
            risk_score += 1
    
    # Check sentence uniformity (low variance = AI-like)
    avg_len, variance, sentences = calculate_sentence_stats(text)
    
    if variance < 20 and len(sentences) > 3:
        flagged.append(f"Niska wariancja długości zdań ({variance:.1f}) - tekst zbyt regularny")
        risk_score += 2
    
    if avg_len > 25:
        flagged.append(f"Bardzo długie średnie zdania ({avg_len:.1f} słów)")
        risk_score += 1
    
    # Check for repetitive structure
    sentence_starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
    start_counts = Counter(sentence_starts)
    common_starts = [(s, c) for s, c in start_counts.items() if c >= 3 and s]
    
    if common_starts:
        for start, count in common_starts:
            flagged.append(f"Powtarzające się początki zdań: '{start}' ({count}x)")
            risk_score += 1
    
    # Unique word ratio (low = formulaic)
    words = text_lower.split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:
            flagged.append(f"Niskie zróżnicowanie słownictwa ({unique_ratio:.2f})")
            risk_score += 1
    
    # Determine risk level
    if risk_score >= 4:
        return "WYSOKI", flagged
    elif risk_score >= 2:
        return "ŚREDNI", flagged
    else:
        return "NISKI", flagged


def test_zerogpt(text: str, retries: int = 3) -> float:
    """Test with ZeroGPT API."""
    for attempt in range(retries):
        try:
            response = requests.post(
                "https://api.zerogpt.com/api/detect/detectText",
                headers={
                    "Content-Type": "application/json",
                    "Origin": "https://www.zerogpt.com",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                },
                json={"input_text": text[:10000]},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("data", {}).get("fakePercentage", 0)
            elif response.status_code == 429:
                time.sleep(10)
                continue
                
        except Exception as e:
            time.sleep(3)
    
    return -1


def find_problematic_sentences(text: str) -> List[str]:
    """Find sentences that are likely to be flagged."""
    problematic = []
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
            
        score = 0
        reasons = []
        
        # Check length
        word_count = len(sentence.split())
        if word_count > 30:
            score += 2
            reasons.append("bardzo długie")
        
        # Check formal patterns
        sentence_lower = sentence.lower()
        formal_starts = ["w związku z", "mając na uwadze", "należy", "istotnym", "podsumowując"]
        for fs in formal_starts:
            if sentence_lower.startswith(fs):
                score += 1
                reasons.append(f"formalne '{fs}'")
        
        # Check passive voice markers
        passive_markers = ["został", "zostało", "zostały", "jest", "są"]
        passive_count = sum(1 for m in passive_markers if m in sentence_lower)
        if passive_count >= 2:
            score += 1
            reasons.append("strona bierna")
        
        if score >= 2:
            problematic.append(f"[{', '.join(reasons)}]: {sentence[:100]}...")
    
    return problematic[:5]  # Top 5


def generate_recommendations(analysis: ChunkAnalysis) -> List[str]:
    """Generate specific recommendations for a chunk."""
    recs = []
    
    if analysis.avg_sentence_length > 25:
        recs.append("Podziel długie zdania na krótsze (cel: 15-20 słów)")
    
    if analysis.sentence_length_variance < 20:
        recs.append("Zwiększ zmienność długości zdań - mieszaj krótkie i długie")
    
    if analysis.risk_level in ["WYSOKI", "ŚREDNI"]:
        recs.append("Zamień formalne konstrukcje na bardziej swobodne")
        recs.append("Dodaj pytania retoryczne lub osobiste komentarze")
    
    if "Powtarzające" in str(analysis.flagged_sentences):
        recs.append("Zróżnicuj początki zdań - unikaj powtórzeń")
    
    return recs


def load_all_chunks(chunks_dir: Path) -> Dict[int, str]:
    """Load all text chunks."""
    chunks = {}
    for f in sorted(chunks_dir.glob("chunk_*.txt")):
        try:
            cid = int(f.stem.split("_")[1])
            with open(f, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Clean
            for noise in [                          "[KOD ŹRÓDŁOWY POMINIĘTY]", "[TABELA]", 
                          "[STRUKTURA KATALOGÓW]", "{}", "{ }", "{\\ }"]:
                text = text.replace(noise, "")
            
            text = text.strip()
            if len(text) > 50:
                chunks[cid] = text
        except:
            continue
    
    return chunks


def main():
    print("=" * 80)
    print("🔬 JSA-LIKE COMPREHENSIVE ANALYSIS")
    print("   Symulacja metodologii polskich systemów antyplagiatowych (perplexity)")
    print("=" * 80)
    
    base_dir = Path(__file__).parent
    chunks_dir = base_dir / "chunks"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load chunks
    chunks = load_all_chunks(chunks_dir)
    print(f"\n📁 Załadowano {len(chunks)} chunków do analizy\n")
    
    analyses = []
    high_risk = []
    medium_risk = []
    
    print("Analizuję chunki... (może potrwać kilka minut)\n")
    print("-" * 80)
    
    for cid in sorted(chunks.keys()):
        text = chunks[cid]
        
        # Calculate stats
        avg_len, variance, sentences = calculate_sentence_stats(text)
        words = text.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        
        # Estimate perplexity risk
        perp_risk, flagged_patterns = estimate_perplexity_risk(text)
        
        # Find problematic sentences
        prob_sentences = find_problematic_sentences(text)
        
        # Test with ZeroGPT
        zerogpt = test_zerogpt(text)
        
        # Determine overall risk
        if perp_risk == "WYSOKI" or zerogpt > 50:
            risk = "🔴 WYSOKI"
        elif perp_risk == "ŚREDNI" or zerogpt > 30:
            risk = "🟡 ŚREDNI"
        else:
            risk = "🟢 NISKI"
        
        analysis = ChunkAnalysis(
            chunk_id=cid,
            text_preview=text[:80] + "...",
            char_count=len(text),
            word_count=len(words),
            avg_sentence_length=avg_len,
            sentence_length_variance=variance,
            unique_word_ratio=unique_ratio,
            estimated_perplexity=perp_risk,
            zerogpt_score=zerogpt,
            risk_level=risk,
            flagged_sentences=flagged_patterns + prob_sentences,
            recommendations=[]
        )
        
        analysis.recommendations = generate_recommendations(analysis)
        analyses.append(analysis)
        
        # Print progress
        zerogpt_str = f"{zerogpt:.0f}%" if zerogpt >= 0 else "błąd"
        print(f"Chunk {cid:2d}: {risk} | Perplexity: {perp_risk:7} | ZeroGPT: {zerogpt_str:5}")
        
        if "WYSOKI" in risk:
            high_risk.append(analysis)
        elif "ŚREDNI" in risk:
            medium_risk.append(analysis)
        
        time.sleep(2)  # Rate limiting
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 PODSUMOWANIE ANALIZY")
    print("=" * 80)
    
    total = len(analyses)
    high_count = len(high_risk)
    medium_count = len(medium_risk)
    low_count = total - high_count - medium_count
    
    print(f"""
📈 STATYSTYKI:
   Przeanalizowano: {total} chunków
   
   🟢 Niskie ryzyko: {low_count} chunków ({low_count/total*100:.0f}%)
   🟡 Średnie ryzyko: {medium_count} chunków ({medium_count/total*100:.0f}%)
   🔴 Wysokie ryzyko: {high_count} chunków ({high_count/total*100:.0f}%)
""")
    
    # High risk details
    if high_risk:
        print("\n" + "=" * 80)
        print("🔴 CHUNKI WYSOKIEGO RYZYKA - WYMAGAJĄ POPRAWY!")
        print("=" * 80)
        
        for a in high_risk:
            print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ CHUNK {a.chunk_id}                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Preview: {a.text_preview[:60]}...
│ ZeroGPT: {a.zerogpt_score:.1f}% | Perplexity risk: {a.estimated_perplexity}
│ Avg sentence length: {a.avg_sentence_length:.1f} | Variance: {a.sentence_length_variance:.1f}
├─────────────────────────────────────────────────────────────────────────────┤
│ PROBLEMY WYKRYTE:""")
            
            for flag in a.flagged_sentences[:5]:
                print(f"│   ⚠️ {flag[:70]}...")
            
            print("├─────────────────────────────────────────────────────────────────────────────┤")
            print("│ REKOMENDACJE:")
            
            for rec in a.recommendations[:3]:
                print(f"│   💡 {rec}")
            
            print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    # Medium risk details
    if medium_risk:
        print("\n" + "=" * 80)
        print("🟡 CHUNKI ŚREDNIEGO RYZYKA - DO WERYFIKACJI")
        print("=" * 80)
        
        for a in medium_risk:
            flags = a.flagged_sentences[:2] if a.flagged_sentences else ["Brak konkretnych flag"]
            print(f"  Chunk {a.chunk_id}: {a.estimated_perplexity} perplexity, ZeroGPT {a.zerogpt_score:.0f}%")
            for f in flags:
                print(f"    → {f[:60]}...")
    
    # Overall verdict
    print("\n" + "=" * 80)
    print("🎯 WERDYKT KOŃCOWY")
    print("=" * 80)
    
    if high_count == 0 and medium_count <= 2:
        print("""
✅ TWOJA PRACA WYGLĄDA BEZPIECZNIE!

Większość chunków ma niskie ryzyko wykrycia przez polskie systemy 
antyplagiatowe. Nadal zalecamy:
1. Sprawdzenie na płatnym narzędziu (plagiat.pl ~ 10 zł)
2. Rozmowę z promotorem przed oddaniem
3. Zachowanie historii edycji
""")
    elif high_count <= 2:
        print(f"""
⚠️ WYMAGA DROBNYCH POPRAWEK

Znaleziono {high_count} chunk(ów) wysokiego ryzyka. Popraw je przed oddaniem:
- Przepisz flagowane fragmenty własnymi słowami
- Zmień strukturę zdań
- Dodaj osobiste komentarze

Chunki do poprawy: {', '.join(str(a.chunk_id) for a in high_risk)}
""")
    else:
        print(f"""
🚨 WYMAGA ZNACZNYCH POPRAWEK

Znaleziono {high_count} chunków wysokiego ryzyka. To poważny problem!
Zalecamy:
1. Przepisanie całego Wstępu
2. Dodanie więcej osobistych komentarzy
3. Zróżnicowanie stylu w całej pracy

Chunki do poprawy: {', '.join(str(a.chunk_id) for a in high_risk)}
""")
    
    # Save detailed results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_chunks": total,
        "high_risk_count": high_count,
        "medium_risk_count": medium_count,
        "low_risk_count": low_count,
        "high_risk_chunks": [a.chunk_id for a in high_risk],
        "medium_risk_chunks": [a.chunk_id for a in medium_risk],
        "detailed_analyses": [asdict(a) for a in analyses]
    }
    
    output_path = results_dir / "jsa_comprehensive_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Szczegółowe wyniki: {output_path}")


if __name__ == "__main__":
    main()
