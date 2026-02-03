"""
Ensemble Scorer - wazone punktowanie z wielu detektorow AI.

Laczy wyniki z roznych detektorow (ZeroGPT, ContentDetector, itd.)
w jeden wazbilansowany werdykt z analiza zgodnosci i pewnosci.

Uzycie: python ensemble_scorer.py
"""

import re
import sys
import time
import statistics
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Callable

from config import DETECTOR_WEIGHTS, THRESHOLDS, RESULTS_DIR
from utils import (
    DetectionResult, load_chunks, call_zerogpt, call_contentdetector,
    setup_logging, save_json_report, save_markdown_report,
    get_cached_result, cache_result,
)


log = setup_logging("ensemble_scorer")


# --- Polskie skroty (nie konczymy zdania na tych kropkach) ---

_PL_ABBREVIATIONS = {
    "np", "dr", "tzw", "wg", "m.in", "prof", "mgr", "inz", "hab",
    "ul", "al", "pl", "tel", "nr", "pkt", "str", "ryc", "tab",
    "rys", "zob", "por", "ok", "tj", "itd", "itp", "art", "ust",
    "zal", "dz", "poz", "rozdz", "przyp", "red", "wyd", "tys",
    "mln", "mld", "godz", "min", "sek", "im", "sw", "jw", "ww",
}

# Wzorzec: skrot + kropka - te kropki nie konczylkoncza zdania
_ABBREV_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(a) for a in _PL_ABBREVIATIONS) + r')\.\s*',
    re.IGNORECASE,
)


# --- Data structures ---

@dataclass
class ChunkVerdict:
    """Werdykt ensemble dla jednego chunka tekstu."""
    chunk_id: int
    text_preview: str
    detector_results: List[DetectionResult]
    weighted_score: float          # 0-100
    agreement: float               # 0-1, jak bardzo detektory sa zgodne
    confidence: str                # "high", "medium", "low"
    classification: str            # "Human", "AI", "Mixed"
    risk_label: str
    flagged_sentences: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["detector_results"] = [r.to_dict() for r in self.detector_results]
        return d


# --- EnsembleScorer ---

class EnsembleScorer:
    """Wazbilansowany ensemble wielu detektorow AI."""

    def __init__(
        self,
        detector_weights: Optional[Dict[str, float]] = None,
        rate_limit_delay: float = 3.0,
    ):
        self.weights = detector_weights or DETECTOR_WEIGHTS
        self.rate_limit_delay = rate_limit_delay
        log.info(f"EnsembleScorer: wagi detektorow = {self.weights}")

    # --- Podzia na zdania ---

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Podziel tekst na zdania, uwzgledniajac polskie skroty."""
        # Zamien skroty na tymczasowe tokeny zeby nie dzielily zdan
        protected = text
        placeholder_map = {}
        for i, match in enumerate(_ABBREV_PATTERN.finditer(text)):
            placeholder = f"__ABBR{i}__"
            placeholder_map[placeholder] = match.group(0)
            protected = protected.replace(match.group(0), placeholder, 1)

        # Podziel na znakach konca zdania
        raw_sentences = re.split(r'(?<=[.!?])\s+', protected)

        # Przywroc skroty
        sentences = []
        for sent in raw_sentences:
            for placeholder, original in placeholder_map.items():
                sent = sent.replace(placeholder, original)
            sent = sent.strip()
            if len(sent) >= 10:  # Minimalna dlugosc zdania
                sentences.append(sent)

        return sentences

    # --- Scoring jednego chunka ---

    def score_chunk(
        self,
        chunk_id: int,
        text: str,
        results: List[DetectionResult],
    ) -> ChunkVerdict:
        """Oblicz wazbilansowany werdykt dla jednego chunka."""

        valid_results = [r for r in results if r.is_valid]

        if not valid_results:
            return ChunkVerdict(
                chunk_id=chunk_id,
                text_preview=text[:120] + "..." if len(text) > 120 else text,
                detector_results=results,
                weighted_score=0.0,
                agreement=0.0,
                confidence="low",
                classification="Brak danych",
                risk_label="Brak danych",
                flagged_sentences=[],
            )

        # Oblicz wazbilansowana srednia
        weighted_score = self._weighted_average(valid_results)

        # Zgodnosc detektorow
        agreement = self._compute_agreement(valid_results)

        # Pewnosc
        confidence = self._compute_confidence(valid_results, agreement)

        # Klasyfikacja
        classification = self._classify(weighted_score)

        # Etykieta ryzyka
        risk_label = THRESHOLDS.label_pl(weighted_score)

        preview = text[:120] + "..." if len(text) > 120 else text

        return ChunkVerdict(
            chunk_id=chunk_id,
            text_preview=preview,
            detector_results=results,
            weighted_score=round(weighted_score, 2),
            agreement=round(agreement, 3),
            confidence=confidence,
            classification=classification,
            risk_label=risk_label,
            flagged_sentences=[],
        )

    def _weighted_average(self, results: List[DetectionResult]) -> float:
        """Oblicz wazbilansowana srednia z wynikow detektorow."""
        total_weight = 0.0
        weighted_sum = 0.0

        for r in results:
            w = self.weights.get(r.detector, None)
            if w is None:
                # Rowna waga jesli detektor nie jest w slowniku
                w = 1.0 / len(results)
            weighted_sum += r.ai_probability * w
            total_weight += w

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _compute_agreement(self, results: List[DetectionResult]) -> float:
        """Zgodnosc detektorow: 1 - (std_dev / 50), zakres [0, 1]."""
        if len(results) < 2:
            return 0.5  # Nie mozna ocenic z jednym detektorem

        scores = [r.ai_probability for r in results]
        std = statistics.stdev(scores)
        agreement = 1.0 - (std / 50.0)
        return max(0.0, min(1.0, agreement))

    def _compute_confidence(
        self,
        results: List[DetectionResult],
        agreement: float,
    ) -> str:
        """Okresl poziom pewnosci."""
        n = len(results)
        if agreement > 0.7 and n >= 3:
            return "high"
        if n >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _classify(score: float) -> str:
        """Klasyfikuj na podstawie score."""
        if score < 30:
            return "Human"
        if score > 70:
            return "AI"
        return "Mixed"

    # --- Scoring poszczegolnych zdan ---

    def score_sentences(
        self,
        text: str,
        detector_func: Callable[[str], float],
        threshold: float = 50.0,
    ) -> List[dict]:
        """Oceniekazbde zdanie osobno, zwroc oflagowane zdania (> threshold)."""
        sentences = self.split_sentences(text)
        flagged = []

        for i, sent in enumerate(sentences):
            if len(sent) < 20:
                continue  # Za krotkie zdanie, pomijamy

            try:
                score = detector_func(sent)
            except Exception as e:
                log.warning(f"Blad scorowania zdania {i}: {e}")
                score = -1.0

            if score > threshold:
                flagged.append({
                    "sentence_idx": i,
                    "sentence": sent,
                    "ai_score": round(score, 2),
                })

            # Rate limiting miedzy zdaniami
            time.sleep(0.5)

        return flagged

    # --- Batch processing ---

    def score_all_chunks(
        self,
        chunks: Dict[int, str],
        detectors: Dict[str, Callable[[str], float]],
    ) -> List[ChunkVerdict]:
        """Przetwarzaj wszystkie chunki ze wszystkimi detektorami."""
        verdicts = []
        total = len(chunks)

        for idx, (chunk_id, text) in enumerate(sorted(chunks.items()), 1):
            log.info(f"Przetwarzam chunk {chunk_id} ({idx}/{total})...")

            results = []
            for det_name, det_func in detectors.items():
                # Sprawdz cache
                cached = get_cached_result(det_name, chunk_id)
                if cached is not None:
                    log.info(f"  {det_name}: wynik z cache ({cached.ai_probability:.1f}%)")
                    results.append(cached)
                    continue

                # Wywolaj detektor
                try:
                    ai_pct = det_func(text)
                    if ai_pct < 0:
                        result = DetectionResult(
                            chunk_id=chunk_id,
                            detector=det_name,
                            ai_probability=-1,
                            human_probability=-1,
                            classification="error",
                            error="Detektor zwrocil blad",
                        )
                    else:
                        classification = self._classify(ai_pct)
                        result = DetectionResult(
                            chunk_id=chunk_id,
                            detector=det_name,
                            ai_probability=round(ai_pct, 2),
                            human_probability=round(100 - ai_pct, 2),
                            classification=classification,
                        )
                    log.info(f"  {det_name}: {result.ai_probability:.1f}% AI")
                except Exception as e:
                    log.error(f"  {det_name}: blad - {e}")
                    result = DetectionResult(
                        chunk_id=chunk_id,
                        detector=det_name,
                        ai_probability=-1,
                        human_probability=-1,
                        classification="error",
                        error=str(e),
                    )

                # Zapisz do cache
                cache_result(result)
                results.append(result)

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            verdict = self.score_chunk(chunk_id, text, results)
            verdicts.append(verdict)

        log.info(f"Zakonczono scoring {len(verdicts)} chunkow.")
        return verdicts

    # --- Generowanie raportu ---

    def generate_report(self, verdicts: List[ChunkVerdict]) -> str:
        """Generuj raport markdown z wynikami ensemble."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        valid_verdicts = [v for v in verdicts if v.weighted_score >= 0]
        if not valid_verdicts:
            return "# Raport Ensemble\n\nBrak prawidlowych wynikow do analizy.\n"

        # Ogolne statystyki
        all_scores = [v.weighted_score for v in valid_verdicts]
        overall_avg = statistics.mean(all_scores) if all_scores else 0
        overall_risk = THRESHOLDS.label_pl(overall_avg)
        overall_emoji = THRESHOLDS.emoji(overall_avg)

        flagged_chunks = [v for v in valid_verdicts if v.weighted_score > 50]
        safe_chunks = [v for v in valid_verdicts if v.weighted_score <= 50]

        lines = []
        lines.append(f"# Raport Ensemble - Detekcja AI w tekscie")
        lines.append(f"")
        lines.append(f"Data: {now}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # --- Podsumowanie ---
        lines.append(f"## Podsumowanie")
        lines.append(f"")
        lines.append(f"| Metryka | Wartosc |")
        lines.append(f"|---------|---------|")
        lines.append(f"| Liczba przeanalizowanych chunkow | {len(valid_verdicts)} |")
        lines.append(f"| Sredni wazony wynik AI | {overall_avg:.1f}% |")
        lines.append(f"| Poziom ryzyka | {overall_emoji} {overall_risk} |")
        lines.append(f"| Chunki oflagowane (>50% AI) | {len(flagged_chunks)} |")
        lines.append(f"| Chunki bezpieczne (<=50% AI) | {len(safe_chunks)} |")
        lines.append(f"")

        # --- Klasyfikacja ogolna ---
        if overall_avg < 30:
            lines.append(f"> **Wniosek:** Tekst wyglada na napisany przez czlowieka. {overall_emoji}")
        elif overall_avg > 70:
            lines.append(f"> **Wniosek:** Tekst wykazuje silne cechy generowania przez AI. {overall_emoji}")
        else:
            lines.append(f"> **Wniosek:** Tekst jest mieszany - zawiera fragmenty ludzkie i potencjalnie AI. {overall_emoji}")
        lines.append(f"")

        # --- Tabela chunkow ---
        lines.append(f"## Wyniki per chunk")
        lines.append(f"")

        # Zbierz nazwy detektorow
        all_detectors = set()
        for v in valid_verdicts:
            for r in v.detector_results:
                all_detectors.add(r.detector)
        all_detectors = sorted(all_detectors)

        # Naglowek tabeli
        det_cols = " | ".join(all_detectors)
        header = f"| Chunk | {det_cols} | Wazbilansowany | Zgodnosc | Pewnosc | Klasyfikacja |"
        separator = "|" + "|".join(["---"] * (len(all_detectors) + 5)) + "|"
        lines.append(header)
        lines.append(separator)

        for v in sorted(valid_verdicts, key=lambda x: x.chunk_id):
            det_scores = {}
            for r in v.detector_results:
                if r.is_valid:
                    det_scores[r.detector] = f"{r.ai_probability:.1f}%"
                else:
                    det_scores[r.detector] = "blad"

            det_vals = " | ".join(det_scores.get(d, "-") for d in all_detectors)
            emoji = THRESHOLDS.emoji(v.weighted_score)

            row = (
                f"| {v.chunk_id:02d} "
                f"| {det_vals} "
                f"| {emoji} {v.weighted_score:.1f}% "
                f"| {v.agreement:.2f} "
                f"| {v.confidence} "
                f"| {v.classification} |"
            )
            lines.append(row)

        lines.append(f"")

        # --- Oflagowane chunki ---
        if flagged_chunks:
            lines.append(f"## Oflagowane chunki (>50% AI)")
            lines.append(f"")
            for v in sorted(flagged_chunks, key=lambda x: -x.weighted_score):
                emoji = THRESHOLDS.emoji(v.weighted_score)
                lines.append(f"### Chunk {v.chunk_id:02d} - {emoji} {v.weighted_score:.1f}% AI")
                lines.append(f"")
                lines.append(f"- **Klasyfikacja:** {v.classification}")
                lines.append(f"- **Ryzyko:** {v.risk_label}")
                lines.append(f"- **Zgodnosc detektorow:** {v.agreement:.2f}")
                lines.append(f"- **Pewnosc:** {v.confidence}")
                lines.append(f"- **Podglad tekstu:** {v.text_preview}")
                lines.append(f"")

                # Wyniki poszczegolnych detektorow
                for r in v.detector_results:
                    if r.is_valid:
                        lines.append(f"  - {r.detector}: {r.ai_probability:.1f}% AI")
                    else:
                        lines.append(f"  - {r.detector}: blad ({r.error})")
                lines.append(f"")

                # Oflagowane zdania
                if v.flagged_sentences:
                    lines.append(f"**Oflagowane zdania:**")
                    lines.append(f"")
                    for fs in v.flagged_sentences:
                        lines.append(f"  - [{fs['ai_score']:.0f}% AI] {fs['sentence'][:100]}...")
                    lines.append(f"")

        # --- Analiza zgodnosci ---
        lines.append(f"## Analiza zgodnosci detektorow")
        lines.append(f"")
        agreements = [v.agreement for v in valid_verdicts]
        avg_agreement = statistics.mean(agreements) if agreements else 0

        lines.append(f"- **Srednia zgodnosc:** {avg_agreement:.2f}")
        if avg_agreement > 0.7:
            lines.append(f"- Detektory sa **wysoce zgodne** w swoich ocenach.")
        elif avg_agreement > 0.4:
            lines.append(f"- Detektory wykazuja **umiarkowana zgodnosc**.")
        else:
            lines.append(f"- Detektory **znacznie sie roznia** w ocenach - wyniki nalezy interpretowac ostroznie.")
        lines.append(f"")

        # Analiza pewnosci
        conf_counts = {"high": 0, "medium": 0, "low": 0}
        for v in valid_verdicts:
            conf_counts[v.confidence] = conf_counts.get(v.confidence, 0) + 1

        lines.append(f"## Pewnosc wynikow")
        lines.append(f"")
        lines.append(f"| Poziom pewnosci | Liczba chunkow |")
        lines.append(f"|-----------------|----------------|")
        lines.append(f"| Wysoka (high)   | {conf_counts['high']} |")
        lines.append(f"| Srednia (medium) | {conf_counts['medium']} |")
        lines.append(f"| Niska (low)     | {conf_counts['low']} |")
        lines.append(f"")

        # --- Uwagi metodologiczne ---
        lines.append(f"## Uwagi metodologiczne")
        lines.append(f"")
        lines.append(f"- Wyniki oparte na ensemble {len(all_detectors)} detektorow z wagami: "
                      + ", ".join(f"{d} ({self.weights.get(d, 'rownowaga')})" for d in all_detectors))
        lines.append(f"- Zgodnosc = 1 - (odchylenie standardowe / 50), zakres [0, 1]")
        lines.append(f"- Detektory AI nie sa w 100% dokladne, szczegolnie dla tekstow w jezyku polskim")
        lines.append(f"- Wynik powyzej 50% nie oznacza pewnosci - jest to sygnal do dalszej analizy")
        lines.append(f"")

        return "\n".join(lines)


# --- Wybor reprezentatywnych chunkow ---

def select_representative_chunks(
    chunks: Dict[int, str],
    count: int = 5,
) -> Dict[int, str]:
    """Wybierz reprezentatywne chunki (poczatek, srodek, koniec + najdluzsze)."""
    if len(chunks) <= count:
        return chunks

    sorted_ids = sorted(chunks.keys())
    n = len(sorted_ids)

    # Strategia: poczatek, 1/4, srodek, 3/4, koniec
    indices = [
        0,
        n // 4,
        n // 2,
        3 * n // 4,
        n - 1,
    ]
    # Usun duplikaty zachowujac kolejnosc
    seen = set()
    unique_indices = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique_indices.append(i)

    selected = {}
    for i in unique_indices[:count]:
        cid = sorted_ids[i]
        selected[cid] = chunks[cid]

    return selected


# --- Main ---

def main():
    """Glowna funkcja: uruchom ensemble scoring."""
    log.info("=" * 60)
    log.info("ENSEMBLE SCORER - Wazbilansowana detekcja AI")
    log.info("=" * 60)

    # Zaladuj chunki
    chunks = load_chunks()
    if not chunks:
        log.error("Nie znaleziono chunkow do analizy!")
        sys.exit(1)

    log.info(f"Zaladowano {len(chunks)} chunkow.")

    # Wybierz 5 reprezentatywnych chunkow
    sample_chunks = select_representative_chunks(chunks, count=5)
    sample_ids = sorted(sample_chunks.keys())
    log.info(f"Wybrano {len(sample_chunks)} reprezentatywnych chunkow: {sample_ids}")

    # Zdefiniuj detektory
    detectors: Dict[str, Callable[[str], float]] = {
        "ZeroGPT": call_zerogpt,
        "ContentDetector": call_contentdetector,
    }

    # Uruchom ensemble scoring
    scorer = EnsembleScorer()
    verdicts = scorer.score_all_chunks(sample_chunks, detectors)

    # Scoring zdan dla oflagowanych chunkow
    flagged_verdicts = [v for v in verdicts if v.weighted_score > 50]
    if flagged_verdicts:
        log.info(f"Scoring zdan dla {len(flagged_verdicts)} oflagowanych chunkow...")
        for v in flagged_verdicts:
            text = sample_chunks.get(v.chunk_id, "")
            if text:
                flagged_sents = scorer.score_sentences(text, call_zerogpt, threshold=50.0)
                v.flagged_sentences = flagged_sents
                log.info(f"  Chunk {v.chunk_id}: {len(flagged_sents)} oflagowanych zdan")

    # Generuj raport
    report_md = scorer.generate_report(verdicts)

    # Zapisz wyniki
    json_data = {
        "data_analizy": datetime.now().isoformat(),
        "liczba_chunkow": len(sample_chunks),
        "sample_chunk_ids": sample_ids,
        "detektory": list(detectors.keys()),
        "wagi": scorer.weights,
        "werdykty": [v.to_dict() for v in verdicts],
        "podsumowanie": {
            "sredni_wynik": round(
                statistics.mean([v.weighted_score for v in verdicts if v.weighted_score >= 0]),
                2,
            ) if verdicts else 0,
            "oflagowane": len(flagged_verdicts),
            "bezpieczne": len(verdicts) - len(flagged_verdicts),
        },
    }

    save_json_report(json_data, "ensemble_report.json")
    save_markdown_report(report_md, "RAPORT_ENSEMBLE.md")

    # Wydrukuj podsumowanie na konsole
    print()
    print("=" * 60)
    print("  WYNIKI ENSEMBLE SCORING")
    print("=" * 60)
    print()

    for v in sorted(verdicts, key=lambda x: x.chunk_id):
        emoji = THRESHOLDS.emoji(v.weighted_score)
        det_strs = []
        for r in v.detector_results:
            if r.is_valid:
                det_strs.append(f"{r.detector}: {r.ai_probability:.1f}%")
            else:
                det_strs.append(f"{r.detector}: blad")
        det_info = " | ".join(det_strs)
        print(
            f"  Chunk {v.chunk_id:02d}: {emoji} {v.weighted_score:5.1f}% AI "
            f"[{v.classification}] (zgodnosc: {v.agreement:.2f}, pewnosc: {v.confidence})"
        )
        print(f"            {det_info}")

    print()
    avg = statistics.mean([v.weighted_score for v in verdicts]) if verdicts else 0
    overall_emoji = THRESHOLDS.emoji(avg)
    print(f"  Sredni wynik: {overall_emoji} {avg:.1f}% AI - {THRESHOLDS.label_pl(avg)}")
    print(f"  Oflagowane chunki: {len(flagged_verdicts)}/{len(verdicts)}")
    print()
    print(f"  Raporty zapisano w: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
