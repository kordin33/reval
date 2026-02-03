"""
MODERN AI DETECTION 2026
Uses latest detectors that can identify GPT-5, Claude 4.5 Opus, Gemini 3

Supported detectors:
1. Originality.ai - 96.5% GPT-5, 98.4% Claude 4, 99% Gemini 3
2. Copyleaks - 99%+ accuracy, 30+ languages
3. GPTZero - 99% accuracy, low false positives
4. Sapling.ai - Updated for latest models

All require API keys but have free trials or credits.
"""

import requests
import json
import time
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional


@dataclass
class DetectionResult:
    chunk_id: int
    detector: str
    ai_probability: float  # 0-100
    human_probability: float
    classification: str
    details: dict = field(default_factory=dict)
    error: Optional[str] = None


class OriginalityAI:
    """
    Originality.ai - Best for detecting GPT-5, Claude 4, Gemini 3
    Accuracy: GPT-5=96.5%, Claude 4=98.4%, Gemini 3=99%+
    Cost: ~$0.01 per 100 words
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ORIGINALITY_API_KEY")
        self.base_url = "https://api.originality.ai/api/v2/scan/ai"
    
    def scan(self, text: str, chunk_id: int) -> DetectionResult:
        if not self.api_key:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Originality.ai",
                ai_probability=-1,
                human_probability=-1,
                classification="no_api_key",
                error="Get API key at: https://originality.ai/ (20 free credits on signup)"
            )
        
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "X-OAI-API-KEY": self.api_key,
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                },
                json={"content": text},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                ai_score = data.get("score", {}).get("ai", 0) * 100
                original_score = data.get("score", {}).get("original", 0) * 100
                
                return DetectionResult(
                    chunk_id=chunk_id,
                    detector="Originality.ai",
                    ai_probability=ai_score,
                    human_probability=original_score,
                    classification="AI" if ai_score > 50 else "Human",
                    details=data
                )
            else:
                return DetectionResult(
                    chunk_id=chunk_id,
                    detector="Originality.ai",
                    ai_probability=-1,
                    human_probability=-1,
                    classification="error",
                    error=f"HTTP {response.status_code}: {response.text[:100]}"
                )
        except Exception as e:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Originality.ai",
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                error=str(e)
            )


class CopyleaksAI:
    """
    Copyleaks - Enterprise-grade, 99%+ accuracy
    Supports 30+ languages including Polish
    """
    
    def __init__(self, api_key: str = None, email: str = None):
        self.api_key = api_key or os.getenv("COPYLEAKS_API_KEY")
        self.email = email or os.getenv("COPYLEAKS_EMAIL")
        self.auth_url = "https://id.copyleaks.com/v3/account/login/api"
        self.detect_url = "https://api.copyleaks.com/v2/writer-detector"
    
    def scan(self, text: str, chunk_id: int) -> DetectionResult:
        if not self.api_key or not self.email:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Copyleaks",
                ai_probability=-1,
                human_probability=-1,
                classification="no_api_key",
                error="Get API at https://copyleaks.com/ai-content-detector (5 free scans)"
            )
        
        # Copyleaks requires OAuth flow - return placeholder
        return DetectionResult(
            chunk_id=chunk_id,
            detector="Copyleaks",
            ai_probability=-1,
            human_probability=-1,
            classification="requires_oauth",
            error="Copyleaks requires OAuth2 - use web interface"
        )


class GPTZeroAPI:
    """
    GPTZero - 99% accuracy, low false positives
    Best for educational content
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GPTZERO_API_KEY")
        self.url = "https://api.gptzero.me/v2/predict/text"
    
    def scan(self, text: str, chunk_id: int) -> DetectionResult:
        if not self.api_key:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="GPTZero",
                ai_probability=-1,
                human_probability=-1,
                classification="no_api_key",
                error="Get API at https://gptzero.me (free tier available)"
            )
        
        try:
            response = requests.post(
                self.url,
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json"
                },
                json={"document": text},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if "documents" in data and data["documents"]:
                    doc = data["documents"][0]
                    ai_prob = doc.get("completely_generated_prob", 0) * 100
                    
                    return DetectionResult(
                        chunk_id=chunk_id,
                        detector="GPTZero",
                        ai_probability=ai_prob,
                        human_probability=100 - ai_prob,
                        classification=doc.get("predicted_class", "unknown"),
                        details={
                            "perplexity": doc.get("average_generated_prob"),
                            "burstiness": doc.get("burstiness")
                        }
                    )
            
            return DetectionResult(
                chunk_id=chunk_id,
                detector="GPTZero",
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                error=f"Unexpected response"
            )
        except Exception as e:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="GPTZero",
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                error=str(e)
            )


class SaplingAI:
    """
    Sapling.ai - Updated for latest models, sentence-level detection
    Free tier: 2000 chars/month
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SAPLING_API_KEY")
        self.url = "https://api.sapling.ai/api/v1/aidetect"
    
    def scan(self, text: str, chunk_id: int) -> DetectionResult:
        if not self.api_key:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Sapling",
                ai_probability=-1,
                human_probability=-1,
                classification="no_api_key",
                error="Get API at https://sapling.ai/user/settings (free tier)"
            )
        
        try:
            response = requests.post(
                self.url,
                json={
                    "key": self.api_key,
                    "text": text,
                    "sent_scores": True
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                score = data.get("score", 0)
                
                flagged_sentences = []
                for sent in data.get("sentence_scores", []):
                    if sent.get("score", 0) > 0.5:
                        flagged_sentences.append({
                            "text": sent["sentence"][:80],
                            "score": sent["score"]
                        })
                
                return DetectionResult(
                    chunk_id=chunk_id,
                    detector="Sapling",
                    ai_probability=score * 100,
                    human_probability=(1 - score) * 100,
                    classification="AI" if score > 0.5 else "Human",
                    details={"flagged_sentences": flagged_sentences[:3]}
                )
            else:
                return DetectionResult(
                    chunk_id=chunk_id,
                    detector="Sapling",
                    ai_probability=-1,
                    human_probability=-1,
                    classification="error",
                    error=f"HTTP {response.status_code}"
                )
        except Exception as e:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Sapling",
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                error=str(e)
            )


def load_chunks(chunks_dir: Path) -> List[tuple]:
    """Load and clean text chunks."""
    chunks = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.txt")):
        chunk_id = int(chunk_file.stem.split("_")[1])
        with open(chunk_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean LaTeX artifacts
        for noise in ["[KOD ŹRÓDŁOWY POMINIĘTY]",
                      "[STRUKTURA KATALOGÓW]", "[TABELA]", "{}", "{ }", "{\\ }"]:
            text = text.replace(noise, "")
        text = text.strip()
        
        if len(text) > 100:  # Skip very short chunks
            chunks.append((chunk_id, text))
    
    return chunks


def generate_report(results: List[DetectionResult], output_path: Path) -> str:
    """Generate comprehensive report."""
    
    lines = [
        "# 🔍 RAPORT DETEKCJI AI - NOWOCZESNE MODELE 2026",
        "",
        f"**Data:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## ℹ️ Użyte detektory",
        "",
        "| Detektor | Dokładność | Wykrywa |",
        "|----------|------------|---------|",
        "| Originality.ai | 96-99% | GPT-5, Claude 4.5, Gemini 3 |",
        "| GPTZero | 99% | GPT-5, Claude, Gemini |",
        "| Sapling | 95%+ | Najnowsze LLM |",
        "| Copyleaks | 99%+ | Wszystkie główne modele |",
        "",
        "---",
        ""
    ]
    
    # Group by detector
    by_detector = {}
    for r in results:
        if r.detector not in by_detector:
            by_detector[r.detector] = []
        by_detector[r.detector].append(r)
    
    # Check which detectors worked
    working_detectors = []
    for det, res_list in by_detector.items():
        valid = [r for r in res_list if r.ai_probability >= 0]
        if valid:
            working_detectors.append(det)
            avg = sum(r.ai_probability for r in valid) / len(valid)
            lines.append(f"### ✅ {det}: średnia {avg:.1f}% AI")
        else:
            error = res_list[0].error if res_list else "Unknown"
            lines.append(f"### ⚠️ {det}: {error}")
    
    lines.append("")
    
    # If we have results, show detailed table
    valid_results = [r for r in results if r.ai_probability >= 0]
    
    if valid_results:
        # Calculate overall
        overall_avg = sum(r.ai_probability for r in valid_results) / len(valid_results)
        
        if overall_avg < 30:
            verdict = "✅ NISKIE RYZYKO - Tekst wygląda na ludzki"
        elif overall_avg < 50:
            verdict = "⚠️ UMIARKOWANE RYZYKO - Możliwe fałszywe alarmy"
        elif overall_avg < 70:
            verdict = "🔶 PODWYŻSZONE RYZYKO - Rozważ edycję"
        else:
            verdict = "🚨 WYSOKIE RYZYKO - Wymaga przepisania"
        
        lines.append("## 📊 PODSUMOWANIE")
        lines.append("")
        lines.append(f"### {verdict}")
        lines.append(f"**Średnia ze wszystkich testów:** {overall_avg:.1f}% AI")
        lines.append("")
        
        # Detailed per-chunk table
        lines.append("## 📝 WYNIKI PER CHUNK")
        lines.append("")
        
        header = "| Chunk |"
        divider = "|-------|"
        for det in working_detectors:
            header += f" {det[:12]} |"
            divider += "-------------|"
        header += " Średnia |"
        divider += "---------|"
        
        lines.append(header)
        lines.append(divider)
        
        # Group by chunk
        by_chunk = {}
        for r in valid_results:
            if r.chunk_id not in by_chunk:
                by_chunk[r.chunk_id] = {}
            by_chunk[r.chunk_id][r.detector] = r
        
        for cid in sorted(by_chunk.keys()):
            row = f"| {cid:5} |"
            vals = []
            for det in working_detectors:
                if det in by_chunk[cid]:
                    p = by_chunk[cid][det].ai_probability
                    vals.append(p)
                    emoji = "🤖" if p > 50 else "👤"
                    row += f" {emoji}{p:.0f}% |"
                else:
                    row += " - |"
            
            if vals:
                avg = sum(vals) / len(vals)
                row += f" {avg:.0f}% |"
            else:
                row += " - |"
            
            lines.append(row)
        
        # Flagged chunks
        flagged = []
        for cid, chunk_res in by_chunk.items():
            vals = [r.ai_probability for r in chunk_res.values()]
            if vals:
                avg = sum(vals) / len(vals)
                if avg > 60:
                    flagged.append((cid, avg))
        
        if flagged:
            lines.append("")
            lines.append("## ⚠️ FRAGMENTY DO POPRAWY (>60% AI)")
            lines.append("")
            for cid, avg in sorted(flagged, key=lambda x: -x[1]):
                lines.append(f"- **Chunk {cid}**: {avg:.0f}% - wymaga przeredagowania")
    
    else:
        lines.append("")
        lines.append("## ❌ BRAK WYNIKÓW")
        lines.append("")
        lines.append("Żaden detektor nie zwrócił wyników. Ustaw klucze API:")
        lines.append("")
        lines.append("```powershell")
        lines.append("$env:ORIGINALITY_API_KEY = \"twoj_klucz\"  # https://originality.ai")
        lines.append("$env:GPTZERO_API_KEY = \"twoj_klucz\"      # https://gptzero.me")
        lines.append("$env:SAPLING_API_KEY = \"twoj_klucz\"      # https://sapling.ai")
        lines.append("```")
    
    # Recommendations
    lines.append("")
    lines.append("## 💡 REKOMENDACJE")
    lines.append("")
    lines.append("""
### Jeśli wyniki są wysokie (>50%):

1. **Przepisz flagowane chunki** własnymi słowami
2. **Dodaj osobiste komentarze**: "Podczas implementacji napotkałem..."
3. **Zróżnicuj styl** - krótkie i długie zdania
4. **Dodaj konkretne szczegóły** - daty, nazwy, numery
5. **Użyj kolokwializmów** gdzie pasuje

### Pamiętaj:
- Detektory AI NIE SĄ 100% dokładne
- Tekst techniczny/akademicki często daje fałszywe alarmy
- W razie wątpliwości - przygotuj dowody procesu pisania
""")
    
    report = "\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


def main():
    print("=" * 60)
    print("🔍 MODERN AI DETECTION 2026")
    print("   Wykrywa: GPT-5, Claude 4.5 Opus, Gemini 3")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    chunks_dir = base_dir / "chunks"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load chunks
    chunks = load_chunks(chunks_dir)
    print(f"\n📁 Załadowano {len(chunks)} chunków")
    
    # Initialize detectors
    detectors = [
        ("Originality.ai", OriginalityAI()),
        ("GPTZero", GPTZeroAPI()),
        ("Sapling", SaplingAI()),
        ("Copyleaks", CopyleaksAI()),
    ]
    
    # Show API key status
    print("\n📋 Status kluczy API:")
    for name, det in detectors:
        has_key = hasattr(det, 'api_key') and det.api_key
        status = "✅" if has_key else "❌"
        print(f"   {status} {name}")
    
    # Sample chunks
    sample_ids = [1, 5, 10, 15, 20]
    sample_chunks = [(cid, text) for cid, text in chunks if cid in sample_ids]
    print(f"\n🔬 Testuję {len(sample_chunks)} próbek na {len(detectors)} detektorach...")
    
    results = []
    
    for chunk_id, text in sample_chunks:
        print(f"\n  📄 Chunk {chunk_id}:")
        
        for name, detector in detectors:
            result = detector.scan(text, chunk_id)
            results.append(result)
            
            if result.ai_probability >= 0:
                emoji = "🤖" if result.ai_probability > 50 else "👤"
                print(f"     {name}: {emoji} {result.ai_probability:.0f}% AI")
            elif "no_api_key" in result.classification:
                print(f"     {name}: ⚠️ brak klucza API")
            else:
                print(f"     {name}: ❌ {result.error[:40]}...")
        
        time.sleep(0.5)
    
    # Generate report
    print("\n" + "=" * 60)
    
    report_path = results_dir / "NOWOCZESNY_RAPORT_AI.md"
    report = generate_report(results, report_path)
    
    # Save raw
    raw_path = results_dir / "modern_detection_raw.json"
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Raport: {report_path}")
    print(f"📄 Raw: {raw_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
