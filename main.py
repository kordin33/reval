"""
AI Text Detection Academic - Centralny punkt wejscia.

Uzycie:
    python main.py --detect              # Detekcja AI na chunkach
    python main.py --evaluate            # Ewaluacja dokladnosci detektorow
    python main.py --analyze-style       # Analiza stylu + sugestie poprawek
    python main.py --binoculars          # Tylko Binoculars (lokalna, GPU)
    python main.py --full                # Wszystko
    python main.py --detect --chunks 1 5 10  # Konkretne chunki
"""

import argparse
import sys
import time
from pathlib import Path

from config import BASE_DIR, RESULTS_DIR, THRESHOLDS
from utils import (
    load_chunks, call_zerogpt, call_contentdetector,
    setup_logging, save_json_report, save_markdown_report,
    DetectionResult, get_cached_result, cache_result,
)

log = setup_logging("main")


def run_detect(chunk_ids=None, use_cache=True):
    """Uruchom detekcje AI na chunkach."""
    print("=" * 60)
    print("DETEKCJA AI")
    print("=" * 60)

    chunks = load_chunks()
    if chunk_ids:
        chunks = {cid: text for cid, text in chunks.items() if cid in chunk_ids}

    if not chunks:
        print("Brak chunkow do analizy.")
        return []

    print(f"Chunki do analizy: {len(chunks)}")

    detectors = [
        ("ZeroGPT", call_zerogpt),
        ("ContentDetector", call_contentdetector),
    ]

    results = []

    for cid in sorted(chunks.keys()):
        text = chunks[cid]
        print(f"\n  Chunk {cid:2d}:", end="")

        for det_name, det_func in detectors:
            # Check cache
            if use_cache:
                cached = get_cached_result(det_name, cid)
                if cached:
                    results.append(cached)
                    emoji = THRESHOLDS.emoji(cached.ai_probability)
                    print(f" {det_name}={emoji}{cached.ai_probability:.0f}%(cache)", end="")
                    continue

            try:
                ai_pct = det_func(text)
                result = DetectionResult(
                    chunk_id=cid,
                    detector=det_name,
                    ai_probability=ai_pct,
                    human_probability=max(0, 100 - ai_pct) if ai_pct >= 0 else -1,
                    classification="AI" if ai_pct > 50 else "Human" if ai_pct >= 0 else "error",
                )
                results.append(result)
                cache_result(result)

                if ai_pct >= 0:
                    emoji = THRESHOLDS.emoji(ai_pct)
                    print(f" {det_name}={emoji}{ai_pct:.0f}%", end="")
                else:
                    print(f" {det_name}=ERR", end="")

            except Exception as e:
                log.error(f"{det_name} chunk {cid}: {e}")
                print(f" {det_name}=ERR", end="")

            time.sleep(2)  # Rate limiting

    print()

    # Summary
    valid = [r for r in results if r.is_valid]
    if valid:
        avg = sum(r.ai_probability for r in valid) / len(valid)
        print(f"\nSrednia: {THRESHOLDS.emoji(avg)} {avg:.1f}% AI")
        print(f"Werdykt: {THRESHOLDS.label_pl(avg)}")

        flagged = set()
        for r in valid:
            if r.ai_probability > 50:
                flagged.add(r.chunk_id)
        if flagged:
            print(f"Flagowane chunki (>50%): {sorted(flagged)}")

    # Save results
    save_json_report(
        [r.to_dict() for r in results],
        "detection_results.json",
    )

    return results


def run_evaluate():
    """Uruchom ewaluacje detektorow."""
    print("=" * 60)
    print("EWALUACJA DETEKTOROW")
    print("=" * 60)

    try:
        from evaluation import EvaluationSuite, POLISH_AI_SAMPLES
        suite = EvaluationSuite()
        chunks = load_chunks()
        human_ids = [5, 10, 12, 14, 15, 16, 18, 20]
        human_chunks = {cid: chunks[cid] for cid in human_ids if cid in chunks}
        suite.run(human_chunks, POLISH_AI_SAMPLES)
    except ImportError as e:
        print(f"Brak modulu evaluation.py: {e}")
    except Exception as e:
        log.error(f"Ewaluacja: {e}")
        raise


def run_binoculars(chunk_ids=None):
    """Uruchom detekcje Binoculars."""
    print("=" * 60)
    print("BINOCULARS (Zero-Shot, Language-Agnostic)")
    print("=" * 60)

    try:
        from binoculars_detector import BinocularsDetector
        detector = BinocularsDetector()
        chunks = load_chunks()
        if chunk_ids:
            chunks = {cid: text for cid, text in chunks.items() if cid in chunk_ids}
        results = detector.detect_batch(chunks)
        for r in results:
            emoji = THRESHOLDS.emoji(r.ai_probability)
            print(f"  Chunk {r.chunk_id:2d}: {emoji} {r.ai_probability:.1f}% AI")
        return results
    except ImportError as e:
        print(f"Brak modulu binoculars_detector.py lub zaleznosci: {e}")
        print("Instaluj: pip install transformers torch")
        return []
    except Exception as e:
        log.error(f"Binoculars: {e}")
        raise


def run_pangram(chunk_ids=None, model_key="roberta"):
    """Uruchom detekcje Pangram EditLens (ICLR 2026)."""
    print("=" * 60)
    print("PANGRAM EDITLENS (ICLR 2026, Open Source)")
    print("=" * 60)

    try:
        from pangram_detector import PangramDetector
        detector = PangramDetector(model_key=model_key)
        chunks = load_chunks()
        if chunk_ids:
            chunks = {cid: text for cid, text in chunks.items() if cid in chunk_ids}
        results = detector.detect_batch(chunks)
        return results
    except ImportError as e:
        print(f"Brak zaleznosci: {e}")
        print("Instaluj: pip install transformers torch")
        print("Zaloguj sie: huggingface-cli login")
        return []
    except Exception as e:
        log.error(f"Pangram: {e}")
        raise


def run_style_analysis(chunk_ids=None, file_path=None):
    """Uruchom analize stylu."""
    print("=" * 60)
    print("ANALIZA STYLU")
    print("=" * 60)

    try:
        from style_analyzer import StyleAnalyzer

        analyzer = StyleAnalyzer()

        if file_path:
            text = Path(file_path).read_text(encoding="utf-8")
            report = analyzer.analyze_text(text, chunk_id=0)
            print(f"\nRyzyko AI: {THRESHOLDS.emoji(report.overall_risk)} {report.overall_risk:.1f}%")
            print(f"Srednia dlugosc zdan: {report.avg_sentence_length:.1f} slow")
            print(f"Unikalne slowa: {report.unique_word_ratio:.2f}")
            print(f"Strona bierna: {report.passive_voice_pct:.0f}%")
            if report.suggestions:
                print("\nSugestie:")
                for s in report.suggestions[:5]:
                    print(f"  - {s}")
            return [report]

        chunks = load_chunks()
        if chunk_ids:
            chunks = {cid: text for cid, text in chunks.items() if cid in chunk_ids}

        reports = []
        for cid in sorted(chunks.keys()):
            report = analyzer.analyze_text(chunks[cid], chunk_id=cid)
            reports.append(report)
            emoji = THRESHOLDS.emoji(report.overall_risk)
            print(f"  Chunk {cid:2d}: {emoji} {report.overall_risk:.1f}% | "
                  f"zdania={report.avg_sentence_length:.0f}sl | "
                  f"unikalne={report.unique_word_ratio:.2f} | "
                  f"bierna={report.passive_voice_pct:.0f}%")

        # Generate report
        md = analyzer.generate_report(reports)
        save_markdown_report(md, "RAPORT_STYL.md")
        return reports

    except ImportError as e:
        print(f"Brak modulu style_analyzer.py: {e}")
        return []
    except Exception as e:
        log.error(f"Analiza stylu: {e}")
        raise


def run_full(chunk_ids=None):
    """Uruchom pelna analize."""
    print()
    print("*" * 60)
    print("  AI TEXT DETECTION - PELNA ANALIZA")
    print("*" * 60)
    print()

    # 1. Detekcja
    detect_results = run_detect(chunk_ids)

    print()

    # 2. Analiza stylu
    style_reports = run_style_analysis(chunk_ids)

    print()

    # 3. Ewaluacja (jesli nie podano konkretnych chunkow)
    if not chunk_ids:
        run_evaluate()

    # Summary
    print()
    print("=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)

    valid_detect = [r for r in detect_results if r.is_valid]
    if valid_detect:
        avg = sum(r.ai_probability for r in valid_detect) / len(valid_detect)
        print(f"\nDetekcja AI: {THRESHOLDS.emoji(avg)} srednia {avg:.1f}%")

    if style_reports:
        avg_risk = sum(r.overall_risk for r in style_reports) / len(style_reports)
        print(f"Ryzyko stylu: {THRESHOLDS.emoji(avg_risk)} srednia {avg_risk:.1f}%")

    print(f"\nRaporty zapisane w: {RESULTS_DIR}")
    print("Gotowe!")


def run_import(input_path: str, chunk_size: int = 2500):
    """Importuj nowa prace (LaTeX lub txt) i podziel na chunki."""
    print("=" * 60)
    print("IMPORT NOWEJ PRACY")
    print("=" * 60)

    from extract_text import process_file
    from config import BASE_DIR

    p = Path(input_path)
    if not p.exists():
        print(f"Plik nie istnieje: {input_path}")
        return

    # Wyczysc cache bo to nowa praca
    cache_dir = RESULTS_DIR / "cache"
    if cache_dir.exists():
        for f in cache_dir.glob("*.json"):
            f.unlink()
        print("Cache wyczyszczony.")

    chunks = process_file(str(p), str(BASE_DIR), chunk_size)
    print(f"\nGotowe! Zaimportowano {len(chunks)} chunkow.")
    print(f"Teraz mozesz uruchomic: python main.py --full")


def main():
    parser = argparse.ArgumentParser(
        description="AI Text Detection Academic - wykrywanie tekstu AI w pracach akademickich",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przyklady:
  python main.py --input praca.tex         # Zaimportuj nowa prace (LaTeX lub txt)
  python main.py --input praca.txt --full  # Import + pelna analiza
  python main.py --full                    # Pelna analiza (na juz zaimportowanych chunkach)
  python main.py --detect                  # Detekcja AI (ZeroGPT + ContentDetector)
  python main.py --detect --chunks 1 5 10  # Konkretne chunki
  python main.py --pangram                 # Pangram EditLens (ICLR 2026, najnowszy!)
  python main.py --binoculars              # Binoculars (GPU, najlepsza dla PL)
  python main.py --evaluate                # Ewaluacja dokladnosci
  python main.py --analyze-style           # Analiza stylu
  python main.py --analyze-style --file chunks/chunk_01.txt  # Jeden plik
        """,
    )

    parser.add_argument("--input", type=str, help="Plik wejsciowy (.tex, .txt) - importuj nowa prace")
    parser.add_argument("--chunk-size", type=int, default=2500, help="Rozmiar chunkow (domyslnie 2500)")
    parser.add_argument("--detect", action="store_true", help="Uruchom detekcje AI")
    parser.add_argument("--evaluate", action="store_true", help="Ewaluacja detektorow")
    parser.add_argument("--analyze-style", action="store_true", help="Analiza stylu")
    parser.add_argument("--pangram", action="store_true", help="Pangram EditLens (ICLR 2026)")
    parser.add_argument("--binoculars", action="store_true", help="Binoculars (GPU)")
    parser.add_argument("--full", action="store_true", help="Pelna analiza")
    parser.add_argument("--chunks", nargs="+", type=int, help="Konkretne chunki do analizy")
    parser.add_argument("--file", type=str, help="Plik do analizy stylu")
    parser.add_argument("--no-cache", action="store_true", help="Ignoruj cache")

    args = parser.parse_args()

    # Import nowej pracy
    if args.input:
        run_import(args.input, args.chunk_size)
        if not any([args.detect, args.evaluate, args.analyze_style, args.binoculars, args.full]):
            return  # Tylko import, bez analizy

    # Default: pokaz help
    actions = [args.input, args.detect, args.evaluate, args.analyze_style,
               args.pangram, args.binoculars, args.full]
    if not any(actions):
        parser.print_help()
        print("\nUzyj --input <plik> aby zaimportowac prace, albo --full dla pelnej analizy.")
        sys.exit(0)

    if args.full:
        run_full(args.chunks)
    else:
        if args.detect:
            run_detect(args.chunks, use_cache=not args.no_cache)
        if args.pangram:
            run_pangram(args.chunks)
        if args.binoculars:
            run_binoculars(args.chunks)
        if args.analyze_style:
            run_style_analysis(args.chunks, args.file)
        if args.evaluate:
            run_evaluate()


if __name__ == "__main__":
    main()
