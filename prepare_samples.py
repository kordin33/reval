"""
Prepare clean text samples for manual AI detection testing.
Generates ready-to-paste text samples for each detector website.
"""

from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    chunks_dir = base_dir / "chunks"
    output_dir = base_dir / "samples_for_testing"
    output_dir.mkdir(exist_ok=True)
    
    # Load selected chunks
    selected = [1, 5, 10, 15, 20, 24]  # Key sections of the thesis
    
    samples = []
    for chunk_id in selected:
        chunk_file = chunks_dir / f"chunk_{chunk_id:02d}.txt"
        if chunk_file.exists():
            with open(chunk_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Clean up any LaTeX artifacts
            text = text.replace("")
            text = text.replace("[KOD ŹRÓDŁOWY POMINIĘTY]", "")
            text = text.replace("[STRUKTURA KATALOGÓW]", "")
            text = text.replace("[TABELA]", "")
            text = text.replace("{}", "")
            text = text.replace("{ }", "")
            text = text.replace("{\\ }", "")
            text = text.strip()
            
            samples.append((chunk_id, text))
            
            # Save individual sample
            sample_file = output_dir / f"sample_{chunk_id:02d}.txt"
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(text)
    
    # Generate combined testing document
    combined = output_dir / "ALL_SAMPLES.txt"
    with open(combined, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PRÓBKI TEKSTU DO TESTOWANIA NA DETEKTORACH AI\n")
        f.write("=" * 60 + "\n\n")
        f.write("Instrukcja:\n")
        f.write("1. Wejdź na stronę detektora (np. copyleaks.com/ai-content-detector)\n")
        f.write("2. Skopiuj tekst poniżej (CTRL+C)\n")
        f.write("3. Wklej na stronie (CTRL+V)\n")
        f.write("4. Zapisz wynik (%AI)\n")
        f.write("5. Powtórz dla każdego fragmentu\n\n")
        
        for chunk_id, text in samples:
            f.write("=" * 60 + "\n")
            f.write(f"FRAGMENT {chunk_id}\n")
            f.write(f"(Długość: {len(text)} znaków)\n")
            f.write("=" * 60 + "\n\n")
            f.write(text)
            f.write("\n\n\n")
    
    print("=" * 60)
    print("📋 PRÓBKI GOTOWE DO TESTOWANIA")
    print("=" * 60)
    print(f"\n📁 Zapisano w: {output_dir}")
    print(f"\nWygenerowano {len(samples)} próbek:")
    for chunk_id, text in samples:
        print(f"  - sample_{chunk_id:02d}.txt ({len(text)} znaków)")
    
    print(f"\n📄 Wszystkie próbki w jednym pliku:")
    print(f"   {combined}")
    
    print("\n" + "=" * 60)
    print("🔗 STRONY DO TESTOWANIA (bez logowania):")
    print("=" * 60)
    print("""
1. Copyleaks: https://copyleaks.com/ai-content-detector
2. ZeroGPT: https://zerogpt.com  
3. Sapling: https://sapling.ai/ai-content-detector
4. Writer: https://writer.com/ai-content-detector/
5. QuillBot: https://quillbot.com/ai-content-detector
6. ContentDetector: https://contentdetector.ai/
    """)
    
    print("=" * 60)
    print("📊 ARKUSZ DO ZAPISYWANIA WYNIKÓW:")
    print("=" * 60)
    print("\n| Fragment | Copyleaks | ZeroGPT | Sapling | Writer | ŚREDNIA |")
    print("|----------|-----------|---------|---------|--------|---------|")
    for chunk_id, _ in samples:
        print(f"|    {chunk_id:02d}    |           |         |         |        |         |")
    
    print("\n" + "=" * 60)
    print("📋 PIERWSZY FRAGMENT DO SKOPIOWANIA:")
    print("=" * 60)
    print("\n" + samples[0][1][:800] + "...")
    print("\n(Pełny tekst w: " + str(output_dir / "sample_01.txt") + ")")


if __name__ == "__main__":
    main()
