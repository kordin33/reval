"""
Simple AI detection test using HuggingFace transformers.
Fallback version with better error handling.
"""

import os
import sys
import json
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("=" * 60)
print("🔍 AI DETECTION - LOKALNE MODELE")
print("=" * 60)

# Check torch installation
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA dostępna: {torch.cuda.is_available()}")
except ImportError:
    print("❌ PyTorch nie zainstalowany")
    sys.exit(1)

try:
    from transformers import pipeline
    print("✅ Transformers załadowane")
except ImportError:
    print("❌ Transformers nie zainstalowane")
    sys.exit(1)


def load_chunks(chunks_dir):
    """Load text chunks."""
    chunks = []
    for chunk_file in sorted(Path(chunks_dir).glob("chunk_*.txt")):
        chunk_id = int(chunk_file.stem.split("_")[1])
        with open(chunk_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean text
        for noise in ["[KOD ŹRÓDŁOWY POMINIĘTY]", 
                      "[STRUKTURA KATALOGÓW]", "[TABELA]", "{}", "{ }", "{\\ }"]:
            text = text.replace(noise, "")
        text = text.strip()
        
        chunks.append((chunk_id, text))
    return chunks


def test_detector(detector_name, model_name, chunks, max_length=512):
    """Test a single detector on all chunks."""
    
    print(f"\n📥 Ładuję model: {model_name}...")
    
    try:
        classifier = pipeline(
            "text-classification",
            model=model_name,
            device=-1,  # CPU
            truncation=True,
            max_length=max_length
        )
        print(f"✅ Model załadowany")
    except Exception as e:
        print(f"❌ Błąd ładowania: {e}")
        return []
    
    results = []
    
    for chunk_id, text in chunks:
        try:
            # Truncate to avoid issues
            text_truncated = text[:1500]
            
            output = classifier(text_truncated)
            
            if isinstance(output, list) and output:
                output = output[0]
            
            label = output.get("label", "").upper()
            score = output.get("score", 0)
            
            # Normalize - different models have different labels
            if any(x in label for x in ["FAKE", "AI", "GPT", "MACHINE", "GENERATED"]):
                ai_prob = score * 100
            elif any(x in label for x in ["REAL", "HUMAN"]):
                ai_prob = (1 - score) * 100
            else:
                ai_prob = score * 100  # assume score is AI probability
            
            results.append({
                "chunk_id": chunk_id,
                "detector": detector_name,
                "ai_probability": ai_prob,
                "label": label,
                "raw_score": score
            })
            
            emoji = "🤖" if ai_prob > 50 else "👤"
            print(f"  Chunk {chunk_id}: {emoji} {ai_prob:.0f}% AI ({label})")
            
        except Exception as e:
            print(f"  Chunk {chunk_id}: ❌ {str(e)[:50]}")
            results.append({
                "chunk_id": chunk_id,
                "detector": detector_name,
                "ai_probability": -1,
                "error": str(e)
            })
    
    return results


def main():
    base_dir = Path(__file__).parent
    chunks_dir = base_dir / "chunks"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load chunks
    chunks = load_chunks(chunks_dir)
    print(f"\n📁 Załadowano {len(chunks)} chunków")
    
    # Sample for speed
    sample_ids = [1, 5, 10, 15, 20]
    sample_chunks = [(cid, text) for cid, text in chunks if cid in sample_ids]
    print(f"📊 Testuję {len(sample_chunks)} próbek")
    
    # Models to try
    models = [
        ("OpenAI-Detector", "openai-community/roberta-base-openai-detector"),
        ("ChatGPT-Detector", "Hello-SimpleAI/chatgpt-detector-roberta"),
    ]
    
    all_results = []
    
    for detector_name, model_name in models:
        results = test_detector(detector_name, model_name, sample_chunks)
        all_results.extend(results)
    
    if not all_results:
        print("\n❌ Nie udało się przeprowadzić żadnych testów")
        return
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 RAPORT")
    print("=" * 60)
    
    # Group by chunk
    by_chunk = {}
    for r in all_results:
        cid = r["chunk_id"]
        if cid not in by_chunk:
            by_chunk[cid] = []
        if r.get("ai_probability", -1) >= 0:
            by_chunk[cid].append(r)
    
    print("\n| Chunk | OpenAI-Detector | ChatGPT-Detector | Średnia |")
    print("|-------|-----------------|------------------|---------|")
    
    chunk_avgs = []
    for chunk_id in sorted(by_chunk.keys()):
        chunk_results = by_chunk[chunk_id]
        
        openai_val = "-"
        chatgpt_val = "-"
        vals = []
        
        for r in chunk_results:
            if r["detector"] == "OpenAI-Detector":
                openai_val = f"{r['ai_probability']:.0f}%"
                vals.append(r['ai_probability'])
            elif r["detector"] == "ChatGPT-Detector":
                chatgpt_val = f"{r['ai_probability']:.0f}%"
                vals.append(r['ai_probability'])
        
        avg = sum(vals) / len(vals) if vals else 0
        chunk_avgs.append(avg)
        
        emoji = "🤖" if avg > 50 else "👤"
        print(f"| {chunk_id:5} | {openai_val:15} | {chatgpt_val:16} | {emoji} {avg:.0f}% |")
    
    # Overall verdict
    overall_avg = sum(chunk_avgs) / len(chunk_avgs) if chunk_avgs else 0
    
    print("\n" + "=" * 60)
    print("🎯 WERDYKT KOŃCOWY")
    print("=" * 60)
    print(f"\n📈 Średnia wszystkich testów: {overall_avg:.1f}% AI")
    
    if overall_avg < 30:
        print("\n✅ NISKIE RYZYKO")
        print("   Twój tekst wygląda na napisany przez człowieka.")
        print("   Prawdopodobnie przejdzie przez większość detektorów.")
    elif overall_avg < 50:
        print("\n⚠️ UMIARKOWANE RYZYKO")
        print("   Niektóre fragmenty mogą być flagowane przez detektory.")
        print("   Rozważ przeredagowanie najbardziej formalnych części.")
    elif overall_avg < 70:
        print("\n🔶 PODWYŻSZONE RYZYKO")
        print("   Tekst ma cechy typowe dla AI.")
        print("   Zalecane przepisanie flagowanych fragmentów.")
    else:
        print("\n🚨 WYSOKIE RYZYKO")
        print("   Większość tekstu wykrywa się jako AI.")
        print("   Konieczne znaczne przepisanie pracy.")
    
    # Recommendations
    print("\n💡 REKOMENDACJE:")
    flagged = [cid for cid, results in by_chunk.items() 
               if sum(r['ai_probability'] for r in results) / len(results) > 60]
    
    if flagged:
        print(f"   Chunki do przepisania: {', '.join(map(str, flagged))}")
    
    print("""
   Jak zmniejszyć wynik:
   1. Dodaj osobiste komentarze: "Podczas moich testów..."
   2. Zróżnicuj długość zdań - krótkie i długie
   3. Użyj kolokwializmów: "Mówiąc wprost...", "Co ciekawe..."
   4. Dodaj konkretne szczegóły: daty, nazwy plików, liczby
    """)
    
    # Save results
    results_file = results_dir / "local_detection_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "overall_avg": overall_avg,
            "results": all_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Wyniki zapisane: {results_file}")


if __name__ == "__main__":
    main()
