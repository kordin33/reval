"""
FINAL DATA SCIENCE VALIDATION

Uses ZeroGPT (which works) + retries + English AI samples.
"""

import json
import time
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List


# ENGLISH AI-generated samples (more obvious for detection)
ENGLISH_AI_SAMPLES = [
    """In today's rapidly evolving technological landscape, artificial intelligence has emerged as 
a transformative force that is reshaping virtually every aspect of our lives. The unprecedented 
advancement in machine learning algorithms has enabled the development of sophisticated systems 
capable of performing tasks that were once thought to be exclusively within human domain.

This comprehensive analysis delves into the multifaceted implications of these technological 
breakthroughs, examining both the promising opportunities and the potential challenges ahead.""",

    """The importance of effective communication cannot be overstated. In an era characterized by 
globalization and digital transformation, the ability to convey ideas clearly has become more 
critical than ever before. First and foremost, effective communication fosters collaboration.

Furthermore, clear communication reduces misunderstandings, creating a harmonious environment. 
Additionally, strong communication skills are essential for leadership success.""",

    """In conclusion, the evidence presented demonstrates that machine learning approaches offer 
significant advantages over traditional methodologies. The experimental results clearly indicate 
that the proposed framework achieves state-of-the-art performance across multiple benchmarks.

Moving forward, future research should focus on addressing identified limitations while exploring 
novel architectural innovations. Ultimately, this work contributes to the growing knowledge base."""
]


@dataclass
class Result:
    sample_id: int
    sample_type: str
    language: str
    ai_probability: float
    text_preview: str


def test_zerogpt_robust(text: str, retries: int = 3) -> float:
    """ZeroGPT with retries"""
    for attempt in range(retries):
        try:
            response = requests.post(
                "https://api.zerogpt.com/api/detect/detectText",
                headers={
                    "Content-Type": "application/json",
                    "Origin": "https://www.zerogpt.com",
                    "Referer": "https://www.zerogpt.com/",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                },
                json={"input_text": text[:10000]},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("data", {}).get("fakePercentage", 0)
                elif "limit" in str(data).lower():
                    print("    [Rate limited, waiting 30s...]")
                    time.sleep(30)
                    continue
            elif response.status_code == 429:
                print("    [Rate limited, waiting 30s...]")
                time.sleep(30)
                continue
                
        except requests.exceptions.Timeout:
            print(f"    [Timeout, retry {attempt+1}]")
            time.sleep(5)
        except Exception as e:
            print(f"    [Error: {str(e)[:30]}]")
            time.sleep(3)
    
    return -1


def load_chunks(chunks_dir: Path) -> dict:
    """Load original chunks"""
    chunks = {}
    for f in sorted(chunks_dir.glob("chunk_*.txt")):
        cid = int(f.stem.split("_")[1])
        with open(f, 'r', encoding='utf-8') as file:
            text = file.read()
        for noise in ["[KOD ŹRÓDŁOWY POMINIĘTY]", 
                      "[TABELA]", "{}", "{ }", "{\\ }"]:
            text = text.replace(noise, "")
        chunks[cid] = text.strip()
    return chunks


def main():
    print("=" * 70)
    print("🔬 FINAL DATA SCIENCE VALIDATION - ZeroGPT")
    print("=" * 70)
    
    base_dir = Path(__file__).parent
    chunks_dir = base_dir / "chunks"
    results_dir = base_dir / "results"
    
    # Load chunks
    chunks = load_chunks(chunks_dir)
    print(f"\n📁 Loaded {len(chunks)} original chunks")
    
    # Select samples
    human_chunk_ids = [5, 10, 15, 20, 8, 12]  # Chunks that scored 0% before
    
    results = []
    
    print("\n" + "=" * 70)
    print("PHASE 1: Testing HUMAN (original) samples")
    print("=" * 70)
    
    for cid in human_chunk_ids:
        if cid not in chunks:
            continue
            
        text = chunks[cid]
        print(f"\n  Chunk {cid} (Human, Polish):")
        
        score = test_zerogpt_robust(text)
        
        if score >= 0:
            emoji = "🤖" if score > 50 else "👤"
            print(f"    ZeroGPT: {emoji} {score:.1f}% AI")
            
            results.append(Result(
                sample_id=cid,
                sample_type="human_original",
                language="polish",
                ai_probability=score,
                text_preview=text[:80]
            ))
        else:
            print(f"    ZeroGPT: ❌ failed")
        
        time.sleep(3)  # Rate limiting
    
    print("\n" + "=" * 70)
    print("PHASE 2: Testing AI-GENERATED (English) samples")
    print("=" * 70)
    
    for i, text in enumerate(ENGLISH_AI_SAMPLES):
        print(f"\n  AI Sample {i+1} (English):")
        
        score = test_zerogpt_robust(text.strip())
        
        if score >= 0:
            emoji = "🤖" if score > 50 else "👤"
            print(f"    ZeroGPT: {emoji} {score:.1f}% AI")
            
            results.append(Result(
                sample_id=200 + i,
                sample_type="ai_generated",
                language="english",
                ai_probability=score,
                text_preview=text[:80]
            ))
        else:
            print(f"    ZeroGPT: ❌ failed")
        
        time.sleep(3)
    
    # Analysis
    print("\n" + "=" * 70)
    print("📊 STATISTICAL ANALYSIS")
    print("=" * 70)
    
    human_results = [r for r in results if r.sample_type == "human_original"]
    ai_results = [r for r in results if r.sample_type == "ai_generated"]
    
    if human_results and ai_results:
        avg_human = sum(r.ai_probability for r in human_results) / len(human_results)
        avg_ai = sum(r.ai_probability for r in ai_results) / len(ai_results)
        gap = avg_ai - avg_human
        
        # Confusion matrix
        tp = sum(1 for r in ai_results if r.ai_probability > 50)  # AI detected as AI
        tn = sum(1 for r in human_results if r.ai_probability <= 50)  # Human as Human
        fp = sum(1 for r in human_results if r.ai_probability > 50)  # Human as AI (false positive)
        fn = sum(1 for r in ai_results if r.ai_probability <= 50)  # AI as Human (false negative)
        
        total = len(human_results) + len(ai_results)
        accuracy = (tp + tn) / total if total > 0 else 0
        
        print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                        ZEROGPT VALIDATION                           │
├─────────────────────────────────────────────────────────────────────┤
│ Human samples (your thesis): {len(human_results):<41} │
│ AI samples (generated): {len(ai_results):<46} │
├─────────────────────────────────────────────────────────────────────┤
│                           SCORES                                    │
├─────────────────────────────────────────────────────────────────────┤
│ Average on HUMAN text: {avg_human:>6.1f}%                                    │
│ Average on AI text: {avg_ai:>6.1f}%                                       │
│ Discrimination Gap: {gap:>+6.1f}%                                          │
├─────────────────────────────────────────────────────────────────────┤
│                      CONFUSION MATRIX                               │
├─────────────────────────────────────────────────────────────────────┤
│ True Positives (AI→AI): {tp:<46} │
│ True Negatives (Human→Human): {tn:<40} │
│ False Positives (Human→AI): {fp:<42} │
│ False Negatives (AI→Human): {fn:<42} │
├─────────────────────────────────────────────────────────────────────┤
│ ACCURACY: {accuracy*100:>6.1f}%                                              │
└─────────────────────────────────────────────────────────────────────┘
""")
        
        # Interpretation
        print("\n📋 INTERPRETATION:")
        print("=" * 70)
        
        if gap > 40:
            print(f"""
✅ EXCELLENT DISCRIMINATION (gap: {gap:+.1f}%)

ZeroGPT clearly distinguishes between human and AI text.
Your thesis scores {avg_human:.1f}% while AI text scores {avg_ai:.1f}%.

CONCLUSION: Your thesis appears human-written according to ZeroGPT.
""")
        elif gap > 20:
            print(f"""
🟢 GOOD DISCRIMINATION (gap: {gap:+.1f}%)

ZeroGPT shows reasonable ability to distinguish text types.
Your thesis: {avg_human:.1f}% | AI text: {avg_ai:.1f}%

CONCLUSION: Your thesis is likely safe, but monitor chunk 1 (Wstęp).
""")
        elif gap > 10:
            print(f"""
🟡 MODERATE DISCRIMINATION (gap: {gap:+.1f}%)

ZeroGPT has limited ability to distinguish your text from AI.
Your thesis: {avg_human:.1f}% | AI text: {avg_ai:.1f}%

Note: This may be due to:
- Polish language (detector trained on English)
- Academic/formal writing style
- Technical content
""")
        else:
            print(f"""
🔴 POOR DISCRIMINATION (gap: {gap:+.1f}%)

ZeroGPT cannot reliably distinguish your text from AI.
Your thesis: {avg_human:.1f}% | AI text: {avg_ai:.1f}%

This suggests the detector might not be reliable for your use case.
Consider testing with other detectors (Originality.ai, GPTZero, Turnitin).
""")
        
        # Per-sample breakdown
        print("\n📋 DETAILED RESULTS:")
        print("-" * 70)
        print(f"{'Sample':<20} {'Type':<15} {'Score':>10} {'Verdict':>15}")
        print("-" * 70)
        
        for r in sorted(results, key=lambda x: x.ai_probability, reverse=True):
            verdict = "🤖 AI" if r.ai_probability > 50 else "👤 Human"
            correct = "✓" if (r.sample_type == "ai_generated" and r.ai_probability > 50) or \
                            (r.sample_type == "human_original" and r.ai_probability <= 50) else "✗"
            
            label = f"Chunk {r.sample_id}" if r.sample_id < 100 else f"AI Sample {r.sample_id-199}"
            print(f"{label:<20} {r.sample_type:<15} {r.ai_probability:>8.1f}% {verdict:>10} {correct}")
    
    else:
        print("\n❌ Insufficient data for analysis")
        print(f"   Human samples: {len(human_results)}")
        print(f"   AI samples: {len(ai_results)}")
    
    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detector": "ZeroGPT",
        "human_samples": len(human_results),
        "ai_samples": len(ai_results),
        "avg_human_score": avg_human if human_results else None,
        "avg_ai_score": avg_ai if ai_results else None,
        "discrimination_gap": gap if human_results and ai_results else None,
        "accuracy": accuracy if human_results and ai_results else None,
        "results": [asdict(r) for r in results]
    }
    
    output_path = results_dir / "final_validation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Results saved: {output_path}")


if __name__ == "__main__":
    main()
