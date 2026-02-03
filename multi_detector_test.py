"""
Multi-detector AI detection test script.
Tests text chunks against multiple AI detection services.

Available free/freemium APIs:
1. Sapling.ai - Free tier available, Python library
2. ZeroGPT - API with free tier  
3. Copyleaks - 5 free scans
4. Writer.com - Free AI detector (web-based)
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import os

@dataclass
class DetectionResult:
    detector_name: str
    chunk_id: int
    ai_probability: float  # 0-100%
    human_probability: float
    classification: str  # "ai", "human", "mixed"
    details: Dict[str, Any]
    error: Optional[str] = None

class SaplingDetector:
    """Sapling.ai AI detector - free tier available."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SAPLING_API_KEY")
        self.url = "https://api.sapling.ai/api/v1/aidetect"
    
    def detect(self, text: str, chunk_id: int) -> DetectionResult:
        if not self.api_key:
            return DetectionResult(
                detector_name="Sapling",
                chunk_id=chunk_id,
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                details={},
                error="No API key provided. Get one at https://sapling.ai/user/settings"
            )
        
        try:
            response = requests.post(
                self.url,
                json={"key": self.api_key, "text": text},
                timeout=30
            )
            data = response.json()
            
            if "score" in data:
                ai_prob = data["score"] * 100
                return DetectionResult(
                    detector_name="Sapling",
                    chunk_id=chunk_id,
                    ai_probability=ai_prob,
                    human_probability=100 - ai_prob,
                    classification="ai" if ai_prob > 50 else "human",
                    details=data
                )
            else:
                return DetectionResult(
                    detector_name="Sapling",
                    chunk_id=chunk_id,
                    ai_probability=-1,
                    human_probability=-1,
                    classification="error",
                    details=data,
                    error=str(data.get("error", "Unknown error"))
                )
        except Exception as e:
            return DetectionResult(
                detector_name="Sapling",
                chunk_id=chunk_id,
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                details={},
                error=str(e)
            )


class ZeroGPTDetector:
    """ZeroGPT detector via RapidAPI or direct API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ZEROGPT_API_KEY")
        self.url = "https://api.zerogpt.com/api/detect/detectText"
    
    def detect(self, text: str, chunk_id: int) -> DetectionResult:
        if not self.api_key:
            return DetectionResult(
                detector_name="ZeroGPT",
                chunk_id=chunk_id,
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                details={},
                error="No API key. Get one at https://zerogpt.com/api"
            )
        
        try:
            headers = {
                "ApiKey": self.api_key,
                "Content-Type": "application/json"
            }
            response = requests.post(
                self.url,
                headers=headers,
                json={"input_text": text},
                timeout=30
            )
            data = response.json()
            
            if data.get("success"):
                ai_prob = data.get("data", {}).get("fakePercentage", 0)
                return DetectionResult(
                    detector_name="ZeroGPT",
                    chunk_id=chunk_id,
                    ai_probability=ai_prob,
                    human_probability=100 - ai_prob,
                    classification="ai" if ai_prob > 50 else "human",
                    details=data.get("data", {})
                )
            else:
                return DetectionResult(
                    detector_name="ZeroGPT",
                    chunk_id=chunk_id,
                    ai_probability=-1,
                    human_probability=-1,
                    classification="error",
                    details=data,
                    error=data.get("message", "API error")
                )
        except Exception as e:
            return DetectionResult(
                detector_name="ZeroGPT",
                chunk_id=chunk_id,
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                details={},
                error=str(e)
            )


class GPTZeroDetector:
    """GPTZero detector."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GPTZERO_API_KEY")
        self.url = "https://api.gptzero.me/v2/predict/text"
    
    def detect(self, text: str, chunk_id: int) -> DetectionResult:
        if not self.api_key:
            return DetectionResult(
                detector_name="GPTZero",
                chunk_id=chunk_id,
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                details={},
                error="No API key. Get one at https://gptzero.me/docs"
            )
        
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            response = requests.post(
                self.url,
                headers=headers,
                json={"document": text},
                timeout=30
            )
            data = response.json()
            
            if "documents" in data:
                doc = data["documents"][0]
                ai_prob = doc.get("completely_generated_prob", 0) * 100
                return DetectionResult(
                    detector_name="GPTZero",
                    chunk_id=chunk_id,
                    ai_probability=ai_prob,
                    human_probability=100 - ai_prob,
                    classification=doc.get("predicted_class", "unknown"),
                    details=doc
                )
            else:
                return DetectionResult(
                    detector_name="GPTZero",
                    chunk_id=chunk_id,
                    ai_probability=-1,
                    human_probability=-1,
                    classification="error",
                    details=data,
                    error=str(data)
                )
        except Exception as e:
            return DetectionResult(
                detector_name="GPTZero",
                chunk_id=chunk_id,
                ai_probability=-1,
                human_probability=-1,
                classification="error",
                details={},
                error=str(e)
            )


class CopyleaksDetector:
    """Copyleaks AI detector - 5 free scans."""
    
    def __init__(self, api_key: str = None, email: str = None):
        self.api_key = api_key or os.getenv("COPYLEAKS_API_KEY")
        self.email = email or os.getenv("COPYLEAKS_EMAIL")
        self.auth_url = "https://id.copyleaks.com/v3/account/login/api"
        self.detect_url = "https://api.copyleaks.com/v2/writer-detector/{scan_id}/check"
    
    def detect(self, text: str, chunk_id: int) -> DetectionResult:
        return DetectionResult(
            detector_name="Copyleaks",
            chunk_id=chunk_id,
            ai_probability=-1,
            human_probability=-1,
            classification="not_tested",
            details={},
            error="Copyleaks requires account setup. Visit https://copyleaks.com/ai-content-detector"
        )


def load_chunks(chunks_dir: Path) -> List[tuple]:
    """Load all text chunks from directory."""
    chunks = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.txt")):
        chunk_id = int(chunk_file.stem.split("_")[1])
        with open(chunk_file, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks.append((chunk_id, text))
    return chunks


def run_detection(chunks: List[tuple], detectors: List, sample_size: int = 5) -> Dict[str, List[DetectionResult]]:
    """Run detection on sample chunks with all available detectors."""
    
    results = {d.__class__.__name__: [] for d in detectors}
    
    # Sample chunks evenly across the document
    if len(chunks) > sample_size:
        step = len(chunks) // sample_size
        sample_indices = [i * step for i in range(sample_size)]
    else:
        sample_indices = list(range(len(chunks)))
    
    sample_chunks = [chunks[i] for i in sample_indices]
    
    print(f"\n{'='*60}")
    print(f"Testing {len(sample_chunks)} chunks across {len(detectors)} detectors")
    print(f"{'='*60}\n")
    
    for detector in detectors:
        detector_name = detector.__class__.__name__
        print(f"\n📊 Testing with {detector_name}...")
        
        for chunk_id, text in sample_chunks:
            print(f"  Chunk {chunk_id}...", end=" ")
            result = detector.detect(text, chunk_id)
            results[detector_name].append(result)
            
            if result.error:
                print(f"❌ Error: {result.error[:50]}...")
            else:
                emoji = "🤖" if result.ai_probability > 50 else "👤"
                print(f"{emoji} AI: {result.ai_probability:.1f}%")
            
            time.sleep(0.5)  # Rate limiting
    
    return results


def aggregate_results(results: Dict[str, List[DetectionResult]]) -> Dict:
    """Aggregate results across all detectors and chunks."""
    
    aggregation = {
        "per_detector": {},
        "per_chunk": {},
        "overall": {
            "avg_ai_probability": 0,
            "flagged_chunks": [],
            "safe_chunks": [],
            "detectors_tested": 0,
            "chunks_tested": 0
        }
    }
    
    all_probs = []
    chunk_probs = {}
    
    for detector_name, detector_results in results.items():
        valid_results = [r for r in detector_results if r.ai_probability >= 0]
        
        if valid_results:
            avg_prob = sum(r.ai_probability for r in valid_results) / len(valid_results)
            aggregation["per_detector"][detector_name] = {
                "avg_ai_probability": avg_prob,
                "classification_counts": {
                    "ai": sum(1 for r in valid_results if r.classification == "ai"),
                    "human": sum(1 for r in valid_results if r.classification == "human"),
                    "mixed": sum(1 for r in valid_results if r.classification == "mixed")
                },
                "chunks_tested": len(valid_results)
            }
            all_probs.extend([r.ai_probability for r in valid_results])
            
            for r in valid_results:
                if r.chunk_id not in chunk_probs:
                    chunk_probs[r.chunk_id] = []
                chunk_probs[r.chunk_id].append(r.ai_probability)
    
    # Per-chunk aggregation
    for chunk_id, probs in chunk_probs.items():
        avg = sum(probs) / len(probs)
        aggregation["per_chunk"][chunk_id] = {
            "avg_ai_probability": avg,
            "min": min(probs),
            "max": max(probs),
            "detectors_count": len(probs)
        }
        
        if avg > 60:
            aggregation["overall"]["flagged_chunks"].append(chunk_id)
        elif avg < 40:
            aggregation["overall"]["safe_chunks"].append(chunk_id)
    
    if all_probs:
        aggregation["overall"]["avg_ai_probability"] = sum(all_probs) / len(all_probs)
        aggregation["overall"]["detectors_tested"] = len([d for d in results if results[d]])
        aggregation["overall"]["chunks_tested"] = len(chunk_probs)
    
    return aggregation


def generate_report(results: Dict, aggregation: Dict, output_path: Path):
    """Generate markdown report with recommendations."""
    
    report = []
    report.append("# 🔍 AI Detection Analysis Report\n")
    report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Overall summary
    report.append("\n## 📊 Overall Summary\n")
    overall = aggregation["overall"]
    avg_prob = overall["avg_ai_probability"]
    
    if avg_prob < 30:
        verdict = "✅ LOW RISK"
        verdict_color = "green"
    elif avg_prob < 50:
        verdict = "⚠️ MODERATE RISK"
        verdict_color = "yellow"
    elif avg_prob < 70:
        verdict = "🔶 ELEVATED RISK"
        verdict_color = "orange"
    else:
        verdict = "🚨 HIGH RISK"
        verdict_color = "red"
    
    report.append(f"**Verdict:** {verdict}\n")
    report.append(f"**Average AI Probability:** {avg_prob:.1f}%\n")
    report.append(f"**Chunks Tested:** {overall['chunks_tested']}\n")
    report.append(f"**Detectors Used:** {overall['detectors_tested']}\n")
    
    # Per-detector breakdown
    report.append("\n## 📈 Results by Detector\n")
    report.append("| Detector | Avg AI % | AI Flagged | Human | Mixed |\n")
    report.append("|----------|----------|------------|-------|-------|\n")
    
    for detector, data in aggregation["per_detector"].items():
        counts = data["classification_counts"]
        report.append(f"| {detector} | {data['avg_ai_probability']:.1f}% | "
                     f"{counts['ai']} | {counts['human']} | {counts['mixed']} |\n")
    
    # Per-chunk breakdown
    report.append("\n## 📝 Results by Chunk\n")
    report.append("| Chunk | Avg AI % | Min | Max | Risk Level |\n")
    report.append("|-------|----------|-----|-----|------------|\n")
    
    for chunk_id in sorted(aggregation["per_chunk"].keys()):
        data = aggregation["per_chunk"][chunk_id]
        avg = data["avg_ai_probability"]
        if avg < 30:
            risk = "🟢 Low"
        elif avg < 50:
            risk = "🟡 Moderate"
        elif avg < 70:
            risk = "🟠 Elevated"
        else:
            risk = "🔴 High"
        
        report.append(f"| {chunk_id} | {avg:.1f}% | {data['min']:.1f}% | "
                     f"{data['max']:.1f}% | {risk} |\n")
    
    # Flagged chunks
    if overall["flagged_chunks"]:
        report.append("\n## ⚠️ Chunks Requiring Attention\n")
        report.append("The following chunks have elevated AI detection scores and may need rephrasing:\n\n")
        for chunk_id in overall["flagged_chunks"]:
            report.append(f"- **Chunk {chunk_id}** - Review and consider humanizing\n")
    
    # Recommendations
    report.append("\n## 💡 Recommendations\n")
    
    if avg_prob < 30:
        report.append("Your text appears to be primarily human-written. No major changes needed.\n")
    else:
        report.append("""
### To Reduce AI Detection Scores:

1. **Vary sentence structure** - Mix short and long sentences
2. **Add personal anecdotes** - Include specific examples from your experience
3. **Use colloquial language** - Add informal expressions where appropriate
4. **Include specific details** - Names, dates, specific numbers
5. **Add rhetorical questions** - Engage the reader directly
6. **Use varied vocabulary** - Avoid repetitive phrases
7. **Include typos/corrections** - Small imperfections (optional, controversial)
8. **Reference personal research process** - "When I ran the experiment..."
9. **Add technical jargon inconsistently** - Mix formal and informal terms
10. **Break patterns** - Avoid formulaic introductions and conclusions
""")
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)


def main():
    """Main execution."""
    
    base_dir = Path(__file__).parent
    chunks_dir = base_dir / "chunks"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 Multi-Detector AI Detection Test")
    print("=" * 50)
    
    # Load chunks
    chunks = load_chunks(chunks_dir)
    print(f"Loaded {len(chunks)} chunks from {chunks_dir}")
    
    # Initialize detectors
    detectors = [
        SaplingDetector(),
        ZeroGPTDetector(),
        GPTZeroDetector(),
        CopyleaksDetector(),
    ]
    
    # Check for API keys
    print("\n📋 Detector Status:")
    for d in detectors:
        name = d.__class__.__name__
        has_key = hasattr(d, 'api_key') and d.api_key
        status = "✅ Ready" if has_key else "⚠️ No API key"
        print(f"  {name}: {status}")
    
    # Run detection
    results = run_detection(chunks, detectors, sample_size=5)
    
    # Aggregate results
    aggregation = aggregate_results(results)
    
    # Generate report
    report_path = output_dir / "ai_detection_report.md"
    report = generate_report(results, aggregation, report_path)
    
    # Save raw results
    raw_results_path = output_dir / "raw_results.json"
    with open(raw_results_path, 'w', encoding='utf-8') as f:
        # Convert dataclass to dict for JSON serialization
        serializable = {}
        for detector, res_list in results.items():
            serializable[detector] = [asdict(r) for r in res_list]
        json.dump({
            "results": serializable,
            "aggregation": aggregation
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 50)
    print("📄 Reports generated:")
    print(f"  - {report_path}")
    print(f"  - {raw_results_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
