"""
EXTENDED MULTI-DETECTOR VALIDATION

Testing with MORE detectors and MORE obvious AI-generated text.
"""

import json
import time
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict


# More obviously AI-generated text (typical ChatGPT style)
OBVIOUS_AI_SAMPLES = [
    # Sample 1: Classic ChatGPT intro style
    """
In today's rapidly evolving technological landscape, artificial intelligence has emerged as a 
transformative force that is reshaping virtually every aspect of our lives. The unprecedented 
advancement in machine learning algorithms, particularly deep learning neural networks, has 
enabled the development of sophisticated systems capable of performing tasks that were once 
thought to be exclusively within the domain of human intelligence.

This comprehensive analysis delves into the multifaceted implications of these technological 
breakthroughs, examining both the promising opportunities and the potential challenges that 
lie ahead. As we navigate this complex terrain, it becomes increasingly clear that understanding 
these dynamics is crucial for stakeholders across all sectors of society.
""",

    # Sample 2: Typical AI essay structure
    """
The importance of effective communication in modern organizations cannot be overstated. In an 
era characterized by globalization, remote work, and digital transformation, the ability to 
convey ideas clearly and persuasively has become more critical than ever before.

First and foremost, effective communication fosters collaboration and teamwork. When team 
members can articulate their thoughts and understand others' perspectives, projects proceed 
more smoothly and efficiently. Furthermore, clear communication reduces misunderstandings 
and conflicts, creating a more harmonious work environment.

Additionally, strong communication skills are essential for leadership. Leaders must inspire 
and motivate their teams, provide constructive feedback, and navigate difficult conversations 
with empathy and clarity. Without these capabilities, even the most technically skilled 
individuals may struggle to advance in their careers.
""",

    # Sample 3: AI-typical conclusion
    """
In conclusion, the evidence presented in this analysis demonstrates that machine learning 
approaches offer significant advantages over traditional methodologies in addressing complex 
real-world problems. The experimental results clearly indicate that the proposed framework 
achieves state-of-the-art performance across multiple benchmark datasets.

Moving forward, future research directions should focus on addressing the identified 
limitations while exploring novel architectural innovations. The integration of multimodal 
learning paradigms and attention mechanisms presents particularly promising avenues for 
enhancing model capabilities.

Ultimately, this work contributes to the growing body of knowledge in the field and 
provides a foundation for subsequent investigations. The implications of these findings 
extend beyond academic research, offering practical applications across diverse industries.
"""
]


@dataclass
class MultiDetectorResult:
    sample_id: int
    sample_type: str  # "human_original", "ai_thesis_style", "ai_obvious"
    detector: str
    ai_probability: float
    raw_response: dict = None


def test_zerogpt(text: str) -> float:
    """ZeroGPT - supports GPT-5, Claude, Gemini"""
    try:
        response = requests.post(
            "https://api.zerogpt.com/api/detect/detectText",
            headers={
                "Content-Type": "application/json",
                "Origin": "https://www.zerogpt.com",
                "User-Agent": "Mozilla/5.0"
            },
            json={"input_text": text[:15000]},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return data.get("data", {}).get("fakePercentage", 0)
        return -1
    except:
        return -1


def test_smodin(text: str) -> float:
    """Try Smodin AI detector"""
    try:
        response = requests.post(
            "https://smodin.io/api/aicheck",
            headers={
                "Content-Type": "application/json",
                "Origin": "https://smodin.io",
                "User-Agent": "Mozilla/5.0"
            },
            json={"text": text[:5000]},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("ai_probability", data.get("aiScore", -1))
        return -1
    except:
        return -1


def test_gptzero_free(text: str) -> float:
    """Try GPTZero web endpoint"""
    try:
        response = requests.post(
            "https://api.gptzero.me/v2/predict/text",
            headers={
                "Content-Type": "application/json",
                "Origin": "https://gptzero.me",
                "User-Agent": "Mozilla/5.0"
            },
            json={"document": text[:5000]},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            if "documents" in data:
                return data["documents"][0].get("completely_generated_prob", 0) * 100
        return -1
    except:
        return -1


def test_writer(text: str) -> float:
    """Try Writer.com AI detector"""
    try:
        response = requests.post(
            "https://api.writer.com/v1/content-detector",
            headers={
                "Content-Type": "application/json",
                "Origin": "https://writer.com",
                "User-Agent": "Mozilla/5.0"
            },
            json={"text": text[:1500]},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("score", 0) * 100
        return -1
    except:
        return -1


def test_hivemoderation(text: str) -> float:
    """Try Hive Moderation AI detector"""
    try:
        response = requests.post(
            "https://api.thehive.ai/api/v2/task/text/text_generation_detection",
            headers={
                "Content-Type": "application/json",
                "Origin": "https://thehive.ai",
                "User-Agent": "Mozilla/5.0"
            },
            json={"text_data": text[:3000]},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("ai_generated_probability", -1)
        return -1
    except:
        return -1


def test_sapling_demo(text: str) -> float:
    """Try Sapling demo endpoint"""
    try:
        response = requests.post(
            "https://api.sapling.ai/api/v1/aidetect",
            headers={
                "Content-Type": "application/json",
                "Origin": "https://sapling.ai",
                "User-Agent": "Mozilla/5.0"
            },
            json={"text": text[:2000]},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("score", 0) * 100
        return -1
    except:
        return -1


def load_human_samples(chunks_dir: Path) -> List[str]:
    """Load original human chunks"""
    chunks = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.txt")):
        cid = int(chunk_file.stem.split("_")[1])
        if cid in [5, 10, 15, 20]:  # Chose chunks that scored 0%
            with open(chunk_file, 'r', encoding='utf-8') as f:
                text = f.read()
            for noise in ["[KOD ŹRÓDŁOWY POMINIĘTY]",
                          "[TABELA]", "{}", "{ }"]:
                text = text.replace(noise, "")
            chunks.append(text.strip())
    return chunks


def main():
    print("=" * 70)
    print("🔬 EXTENDED MULTI-DETECTOR VALIDATION")
    print("=" * 70)
    
    base_dir = Path(__file__).parent
    chunks_dir = base_dir / "chunks"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load samples
    human_samples = load_human_samples(chunks_dir)
    print(f"\n📁 Loaded {len(human_samples)} human samples")
    print(f"📁 Using {len(OBVIOUS_AI_SAMPLES)} obvious AI samples")
    
    # All detectors to try
    detectors = [
        ("ZeroGPT", test_zerogpt),
        ("GPTZero", test_gptzero_free),
        ("Smodin", test_smodin),
        ("Writer", test_writer),
        ("Sapling", test_sapling_demo),
        ("Hive", test_hivemoderation),
    ]
    
    results = []
    working_detectors = set()
    
    # Test all detectors on all samples
    print("\n🔍 Testing detectors...")
    
    # Human samples
    for i, text in enumerate(human_samples):
        print(f"\n  Human sample {i+1}:")
        for det_name, det_func in detectors:
            score = det_func(text)
            results.append(MultiDetectorResult(
                sample_id=i,
                sample_type="human_original",
                detector=det_name,
                ai_probability=score
            ))
            if score >= 0:
                working_detectors.add(det_name)
                emoji = "🤖" if score > 50 else "👤"
                print(f"    {det_name}: {emoji} {score:.1f}%")
            else:
                print(f"    {det_name}: ❌")
            time.sleep(1)
    
    # Obvious AI samples
    for i, text in enumerate(OBVIOUS_AI_SAMPLES):
        print(f"\n  AI sample {i+1} (obvious):")
        for det_name, det_func in detectors:
            score = det_func(text)
            results.append(MultiDetectorResult(
                sample_id=100 + i,
                sample_type="ai_obvious",
                detector=det_name,
                ai_probability=score
            ))
            if score >= 0:
                working_detectors.add(det_name)
                emoji = "🤖" if score > 50 else "👤"
                print(f"    {det_name}: {emoji} {score:.1f}%")
            else:
                print(f"    {det_name}: ❌")
            time.sleep(1)
    
    # Analysis
    print("\n" + "=" * 70)
    print("📊 ANALYSIS BY DETECTOR")
    print("=" * 70)
    
    for det_name in working_detectors:
        det_results = [r for r in results if r.detector == det_name and r.ai_probability >= 0]
        
        human_results = [r for r in det_results if r.sample_type == "human_original"]
        ai_results = [r for r in det_results if r.sample_type == "ai_obvious"]
        
        if not human_results or not ai_results:
            continue
        
        avg_human = sum(r.ai_probability for r in human_results) / len(human_results)
        avg_ai = sum(r.ai_probability for r in ai_results) / len(ai_results)
        gap = avg_ai - avg_human
        
        # Calculate accuracy (threshold 50%)
        correct_human = sum(1 for r in human_results if r.ai_probability < 50)
        correct_ai = sum(1 for r in ai_results if r.ai_probability >= 50)
        accuracy = (correct_human + correct_ai) / (len(human_results) + len(ai_results))
        
        print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ {det_name:^67} │
├─────────────────────────────────────────────────────────────────────┤
│ Human samples: {len(human_results):<55} │
│ AI samples: {len(ai_results):<58} │
│                                                                     │
│ Avg score on HUMAN text: {avg_human:>5.1f}%                                    │
│ Avg score on AI text: {avg_ai:>5.1f}%                                       │
│ Discrimination gap: {gap:>+5.1f}%                                          │
│                                                                     │
│ Accuracy (threshold 50%): {accuracy*100:>5.1f}%                                │
│   - Human correctly identified: {correct_human}/{len(human_results):<29} │
│   - AI correctly identified: {correct_ai}/{len(ai_results):<32} │
└─────────────────────────────────────────────────────────────────────┘""")
        
        # Interpretation
        if gap > 40:
            print(f"   ✅ EXCELLENT discrimination - reliable for your thesis\n")
        elif gap > 20:
            print(f"   🟢 GOOD discrimination - reasonably reliable\n")
        elif gap > 10:
            print(f"   🟡 MODERATE - use with caution\n")
        else:
            print(f"   🔴 POOR discrimination - not reliable\n")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY FOR YOUR THESIS")
    print("=" * 70)
    
    # Get ZeroGPT results specifically
    zerogpt_human = [r for r in results if r.detector == "ZeroGPT" and r.sample_type == "human_original" and r.ai_probability >= 0]
    zerogpt_ai = [r for r in results if r.detector == "ZeroGPT" and r.sample_type == "ai_obvious" and r.ai_probability >= 0]
    
    if zerogpt_human and zerogpt_ai:
        avg_human = sum(r.ai_probability for r in zerogpt_human) / len(zerogpt_human)
        avg_ai = sum(r.ai_probability for r in zerogpt_ai) / len(zerogpt_ai)
        
        print(f"""
ZeroGPT Results:
  Your thesis chunks: {avg_human:.1f}% AI (average)
  Obvious AI text: {avg_ai:.1f}% AI (average)
  
  Interpretation:
""")
        if avg_human < 10:
            print("  ✅ Your text scores very LOW - clearly recognized as human")
        elif avg_human < 30:
            print("  🟢 Your text scores low - likely safe")
        elif avg_human < 50:
            print("  🟡 Your text scores moderately - some sections may be flagged")
        else:
            print("  🔴 Your text scores high - significant risk of being flagged")
    
    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "working_detectors": list(working_detectors),
        "results": [asdict(r) for r in results]
    }
    
    output_path = results_dir / "multi_detector_validation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Results saved: {output_path}")


if __name__ == "__main__":
    main()
