"""
Web-based AI detection without browser automation.
Uses direct HTTP requests to free AI detection services where possible.
"""

import requests
import json
import time
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import hashlib


@dataclass  
class DetectionResult:
    chunk_id: int
    detector: str
    ai_probability: float
    classification: str
    details: dict
    error: Optional[str] = None


def test_zerogpt_web(text: str, chunk_id: int) -> DetectionResult:
    """
    Try ZeroGPT through their web API endpoint.
    """
    
    url = "https://api.zerogpt.com/api/detect/detectText"
    
    try:
        # ZeroGPT web interface endpoint
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Origin": "https://www.zerogpt.com",
                "Referer": "https://www.zerogpt.com/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            json={"input_text": text[:15000]},  # ZeroGPT limit
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("success"):
                result_data = data.get("data", {})
                fake_pct = result_data.get("fakePercentage", 0)
                
                return DetectionResult(
                    chunk_id=chunk_id,
                    detector="ZeroGPT",
                    ai_probability=fake_pct,
                    classification="AI" if fake_pct > 50 else "Human",
                    details={
                        "sentences": result_data.get("feedback", [])[:3],
                        "is_human": result_data.get("isHuman"),
                        "is_gpt": result_data.get("isGpt")
                    }
                )
            else:
                return DetectionResult(
                    chunk_id=chunk_id,
                    detector="ZeroGPT",
                    ai_probability=-1,
                    classification="error",
                    details={},
                    error=data.get("message", "API error")
                )
        else:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="ZeroGPT",
                ai_probability=-1,
                classification="error",
                details={},
                error=f"HTTP {response.status_code}"
            )
            
    except Exception as e:
        return DetectionResult(
            chunk_id=chunk_id,
            detector="ZeroGPT",
            ai_probability=-1,
            classification="error",
            details={},
            error=str(e)
        )


def test_writer_web(text: str, chunk_id: int) -> DetectionResult:
    """
    Try Writer.com AI detector through their API.
    """
    
    url = "https://writer.com/api/content-detector"
    
    try:
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Origin": "https://writer.com",
                "Referer": "https://writer.com/ai-content-detector/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            json={"text": text[:1500]},  # Writer limit ~1500 words
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            ai_score = data.get("score", 0) * 100 if isinstance(data.get("score"), float) else 0
            
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Writer.com",
                ai_probability=ai_score,
                classification="AI" if ai_score > 50 else "Human",
                details=data
            )
        else:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Writer.com",
                ai_probability=-1,
                classification="error",
                details={},
                error=f"HTTP {response.status_code} - likely needs browser"
            )
            
    except Exception as e:
        return DetectionResult(
            chunk_id=chunk_id,
            detector="Writer.com",
            ai_probability=-1,
            classification="error",
            details={},
            error=str(e)
        )


def test_contentdetector_web(text: str, chunk_id: int) -> DetectionResult:
    """
    Try ContentDetector.AI
    """
    
    url = "https://contentdetector.ai/api/detectAi"
    
    try:
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Origin": "https://contentdetector.ai",
                "Referer": "https://contentdetector.ai/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            json={"text": text},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            ai_prob = data.get("probability", 0)
            
            return DetectionResult(
                chunk_id=chunk_id,
                detector="ContentDetector.AI",
                ai_probability=ai_prob * 100 if ai_prob <= 1 else ai_prob,
                classification=data.get("label", "unknown"),
                details=data
            )
        else:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="ContentDetector.AI",
                ai_probability=-1,
                classification="error",
                details={},
                error=f"HTTP {response.status_code}"
            )
            
    except Exception as e:
        return DetectionResult(
            chunk_id=chunk_id,
            detector="ContentDetector.AI",
            ai_probability=-1,
            classification="error", 
            details={},
            error=str(e)
        )


def test_sapling_free(text: str, chunk_id: int) -> DetectionResult:
    """
    Sapling has a free demo - try to access it
    """
    
    # Sapling requires API key, but let's check if demo works
    url = "https://api.sapling.ai/api/v1/aidetect"
    
    # Try without API key (might work for demo)
    try:
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Origin": "https://sapling.ai",
                "Referer": "https://sapling.ai/ai-content-detector"
            },
            json={"text": text[:2000]},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            score = data.get("score", 0)
            
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Sapling (demo)",
                ai_probability=score * 100,
                classification="AI" if score > 0.5 else "Human",
                details=data
            )
        else:
            return DetectionResult(
                chunk_id=chunk_id,
                detector="Sapling (demo)",
                ai_probability=-1,
                classification="needs_api_key",
                details={},
                error="Requires API key from sapling.ai"
            )
            
    except Exception as e:
        return DetectionResult(
            chunk_id=chunk_id,
            detector="Sapling (demo)",
            ai_probability=-1,
            classification="error",
            details={},
            error=str(e)
        )


def load_chunks(chunks_dir: Path) -> List[tuple]:
    """Load text chunks."""
    chunks = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.txt")):
        chunk_id = int(chunk_file.stem.split("_")[1])
        with open(chunk_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean
        for noise in ["[KOD ŹRÓDŁOWY POMINIĘTY]",
                      "[STRUKTURA KATALOGÓW]", "[TABELA]", "{}", "{ }", "{\\ }"]:
            text = text.replace(noise, "")
        text = text.strip()
        
        if len(text) > 100:
            chunks.append((chunk_id, text))
    
    return chunks


def main():
    print("=" * 60)
    print("🔍 WEB AI DETECTION (bez API keys)")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    chunks_dir = base_dir / "chunks"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    chunks = load_chunks(chunks_dir)
    print(f"\n📁 Załadowano {len(chunks)} chunków")
    
    # Sample
    sample_ids = [1, 5, 10, 15, 20]
    sample_chunks = [(cid, text) for cid, text in chunks if cid in sample_ids]
    
    print(f"🔬 Testuję {len(sample_chunks)} próbek...")
    
    detectors = [
        ("ZeroGPT", test_zerogpt_web),
        ("Writer.com", test_writer_web),
        ("ContentDetector.AI", test_contentdetector_web),
        ("Sapling", test_sapling_free),
    ]
    
    results = []
    
    for chunk_id, text in sample_chunks:
        print(f"\n📄 Chunk {chunk_id}:")
        
        for name, func in detectors:
            result = func(text, chunk_id)
            results.append(result)
            
            if result.ai_probability >= 0:
                emoji = "🤖" if result.ai_probability > 50 else "👤"
                print(f"   {name}: {emoji} {result.ai_probability:.0f}% AI")
            else:
                print(f"   {name}: ⚠️ {result.error[:40] if result.error else 'error'}...")
            
            time.sleep(1)  # Rate limiting
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PODSUMOWANIE")
    print("=" * 60)
    
    valid_results = [r for r in results if r.ai_probability >= 0]
    
    if valid_results:
        avg_all = sum(r.ai_probability for r in valid_results) / len(valid_results)
        
        print(f"\n✅ Uzyskano {len(valid_results)} wyników")
        print(f"📈 Średnia AI: {avg_all:.1f}%")
        
        if avg_all < 30:
            print("\n🟢 NISKIE RYZYKO - Tekst wygląda na ludzki")
        elif avg_all < 50:
            print("\n🟡 UMIARKOWANE - Możliwe fałszywe alarmy")
        elif avg_all < 70:
            print("\n🟠 PODWYŻSZONE - Rozważ edycję niektórych fragmentów")
        else:
            print("\n🔴 WYSOKIE RYZYKO - Wymaga przepisania")
        
        # Per detector
        print("\n📋 Wyniki per detektor:")
        by_det = {}
        for r in valid_results:
            if r.detector not in by_det:
                by_det[r.detector] = []
            by_det[r.detector].append(r.ai_probability)
        
        for det, probs in by_det.items():
            avg = sum(probs) / len(probs)
            print(f"   {det}: {avg:.0f}% AI (średnia z {len(probs)} testów)")
    
    else:
        print("\n❌ Żaden detektor nie zwrócił wyników")
        print("   Możliwe przyczyny:")
        print("   - Serwisy wymagają CAPTCHA lub autoryzacji")
        print("   - API zostało zmienione")
        print("   - Rate limiting")
        print("\n💡 Spróbuj ręcznie na:")
        print("   → https://zerogpt.com")
        print("   → https://copyleaks.com/ai-content-detector")
        print("   → https://gptzero.me")
    
    # Save
    raw_path = results_dir / "web_detection_results.json"
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Wyniki zapisane: {raw_path}")


if __name__ == "__main__":
    main()
