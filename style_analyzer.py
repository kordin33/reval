"""
Analizator stylu tekstu akademickiego w jezyku polskim.
Wykrywa wzorce typowe dla tekstu generowanego przez AI
i proponuje konkretne poprawki zwiekszajace naturalnosc.

Uzycie:
    python style_analyzer.py                        # analiza wszystkich chunkow
    python style_analyzer.py chunks/chunk_01.txt    # analiza jednego pliku
"""

import re
import sys
import math
import statistics
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from config import (
    FORMAL_PATTERNS_PL, THRESHOLDS, RESULTS_DIR,
    MIN_SENTENCE_LENGTH, MAX_GOOD_SENTENCE_WORDS,
    MIN_GOOD_VARIANCE, MIN_GOOD_UNIQUE_RATIO,
)
from utils import load_chunks, setup_logging, save_markdown_report


log = setup_logging("style_analyzer")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SentenceAnalysis:
    text: str
    word_count: int
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass
class StyleReport:
    chunk_id: int
    text: str
    sentences: List[SentenceAnalysis] = field(default_factory=list)
    overall_risk: float = 0.0
    avg_sentence_length: float = 0.0
    sentence_length_std: float = 0.0
    unique_word_ratio: float = 0.0
    formal_patterns_found: List[str] = field(default_factory=list)
    repetitive_starts: List[tuple] = field(default_factory=list)
    passive_voice_pct: float = 0.0
    suggestions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stale
# ---------------------------------------------------------------------------

# Polskie skroty, przy ktorych kropka NIE konczy zdania
_PL_ABBREVIATIONS = [
    "m.in", "np", "dr", "prof", "tzw", "wg", "ok", "tj",
    "itp", "itd", "mgr", "inz", "hab", "art", "ust",
    "pkt", "str", "tab", "rys", "zob", "por", "wyd",
    "vol", "nr", "s", "r", "w", "t",
]

# Regex do ochrony skrotow przed splitowaniem
_ABBR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _PL_ABBREVIATIONS) + r")\.\s*",
    re.IGNORECASE,
)

# Markery strony biernej
_PASSIVE_MARKERS = [
    r"\bzosta[lł](?:a|o|y|em|am|i)?\b",
    r"\bjest\s+\w+(?:any|ony|eny|ity|yty|ety|owany|ywany|iwany)\b",
    r"\bs[aą]\s+\w+(?:ane|one|ene|ite|yte|ete|owane|ywane|iwane)\b",
    r"\bby[lł](?:a|o|y)?\s+\w+(?:any|ony|eny|ity|yty|owany|ywany)\b",
    r"\bzostanie\b",
    r"\bzostaj[aą]\b",
]
_PASSIVE_RE = [re.compile(p, re.IGNORECASE) for p in _PASSIVE_MARKERS]

# Hedging / ostroznosciowe sformulowania
_HEDGING_PATTERNS = [
    (re.compile(r"mo[zż]na\s+zauwa[zż]y[cć]", re.I), "mozna zauwazyc"),
    (re.compile(r"wydaje\s+si[eę]", re.I), "wydaje sie"),
    (re.compile(r"nale[zż]y\s+stwierdzi[cć]", re.I), "nalezy stwierdzic"),
    (re.compile(r"mo[zż]na\s+stwierdzi[cć]", re.I), "mozna stwierdzic"),
    (re.compile(r"nale[zż]y\s+podkre[sś]li[cć]", re.I), "nalezy podkreslic"),
    (re.compile(r"warto\s+wspomnie[cć]", re.I), "warto wspomniec"),
    (re.compile(r"nale[zż]y\s+wspomnie[cć]", re.I), "nalezy wspomniec"),
]

# Formalne poczatki zdan
_FORMAL_STARTS = [
    re.compile(r"^W\s+zwi[aą]zku\s+z", re.I),
    re.compile(r"^Nale[zż]y\b", re.I),
    re.compile(r"^Maj[aą]c\s+na\s+uwadze", re.I),
    re.compile(r"^Podsumowuj[aą]c", re.I),
    re.compile(r"^Istotnym\s+aspektem", re.I),
    re.compile(r"^Na\s+podstawie\s+przeprowadzon", re.I),
    re.compile(r"^W\s+niniejszej\s+pracy", re.I),
    re.compile(r"^Powy[zż]sze\s+rozwa[zż]ania", re.I),
    re.compile(r"^Zagadnienie\s+to", re.I),
    re.compile(r"^W\s+kontek[sś]cie", re.I),
    re.compile(r"^Analiza\s+wykazuje", re.I),
]

# Tabela zamiennikow
REPLACEMENT_TABLE: List[Tuple[str, str, List[str]]] = [
    ("z punktu widzenia",
     r"z\s+punktu\s+widzenia",
     ["patrz\u0105c na to z perspektywy"]),
    ("nale\u017cy podkre\u015bli\u0107",
     r"nale[z\u017c]y\s+podkre[s\u015b]li[c\u0107]",
     ["warto zauwa\u017cy\u0107", "ciekawe jest to, \u017ce"]),
    ("w niniejszej pracy",
     r"w\s+niniejszej\s+pracy",
     ["w tej pracy", "tutaj pokazuj\u0119"]),
    ("podsumowuj\u0105c, mo\u017cna stwierdzi\u0107",
     r"podsumowuj[a\u0105]c,?\s*mo[z\u017c]na\s+stwierdzi[c\u0107]",
     ["kr\u00f3tko m\u00f3wi\u0105c"]),
    ("w zwi\u0105zku z powy\u017cszym",
     r"w\s+zwi[a\u0105]zku\s+z\s+powy[z\u017c]szym",
     ["dlatego", "w efekcie"]),
    ("maj\u0105c na uwadze",
     r"maj[a\u0105]c\s+na\s+uwadze",
     ["bior\u0105c pod uwag\u0119", "skoro tak"]),
    ("istotnym aspektem jest",
     r"istotnym\s+aspektem\s+jest",
     ["wa\u017cne jest to, \u017ce"]),
    ("wykazano, \u017ce",
     r"wykazano,?\s*[z\u017c]e",
     ["okazuje si\u0119, \u017ce"]),
    ("przeprowadzona analiza",
     r"przeprowadzon[a]?\s+analiz[a]?",
     ["analiza pokaza\u0142a"]),
    ("mo\u017cna zauwa\u017cy\u0107",
     r"mo[z\u017c]na\s+zauwa[z\u017c]y[c\u0107]",
     ["widzimy, \u017ce", "wida\u0107, \u017ce"]),
    ("wydaje si\u0119",
     r"wydaje\s+si[e\u0119]",
     ["wygl\u0105da na to, \u017ce", "chyba"]),
    ("nale\u017cy stwierdzi\u0107",
     r"nale[z\u017c]y\s+stwierdzi[c\u0107]",
     ["trzeba przyzna\u0107", "faktem jest"]),
]


# ---------------------------------------------------------------------------
# StyleAnalyzer
# ---------------------------------------------------------------------------

class StyleAnalyzer:
    """Analizator stylu tekstu akademickiego pod katem wzorcow AI."""

    # ---- public API -------------------------------------------------------

    def analyze_text(self, text: str, chunk_id: int = 0) -> StyleReport:
        """Pelna analiza stylistyczna fragmentu tekstu."""
        sentences_raw = self._split_sentences(text)
        sentence_analyses = [self._analyze_sentence(s) for s in sentences_raw]

        lengths = [sa.word_count for sa in sentence_analyses if sa.word_count > 0]
        avg_len = statistics.mean(lengths) if lengths else 0.0
        std_len = statistics.stdev(lengths) if len(lengths) >= 2 else 0.0

        words = re.findall(r"\w+", text.lower())
        unique_ratio = len(set(words)) / len(words) if words else 1.0

        formal_found = self._check_formal_patterns(text)
        rep_starts = self._check_repetitive_starts(sentences_raw)
        passive_pct = self._calculate_passive_voice_pct(sentences_raw)

        report = StyleReport(
            chunk_id=chunk_id,
            text=text,
            sentences=sentence_analyses,
            avg_sentence_length=round(avg_len, 1),
            sentence_length_std=round(std_len, 1),
            unique_word_ratio=round(unique_ratio, 3),
            formal_patterns_found=formal_found,
            repetitive_starts=rep_starts,
            passive_voice_pct=round(passive_pct, 1),
        )

        report.overall_risk = self._overall_risk(report)
        report.suggestions = self._global_suggestions(report)

        return report

    def compare(
        self,
        original_text: str,
        modified_text: str,
        chunk_id: int = 0,
    ) -> dict:
        """Porownanie oryginalnego i poprawionego tekstu."""
        orig = self.analyze_text(original_text, chunk_id)
        mod = self.analyze_text(modified_text, chunk_id)

        delta = mod.overall_risk - orig.overall_risk

        return {
            "chunk_id": chunk_id,
            "oryginalne_ryzyko": round(orig.overall_risk, 1),
            "nowe_ryzyko": round(mod.overall_risk, 1),
            "delta": round(delta, 1),
            "poprawa": f"Poprawa: {delta:+.0f}% ryzyka AI"
                       if delta < 0
                       else f"Pogorszenie: +{delta:.0f}% ryzyka AI"
                       if delta > 0
                       else "Bez zmian",
            "szczegoly_przed": {
                "sr_dlug_zdania": orig.avg_sentence_length,
                "odch_dlug_zdania": orig.sentence_length_std,
                "unikat_slow": orig.unique_word_ratio,
                "formalne_wzorce": len(orig.formal_patterns_found),
                "strona_bierna": orig.passive_voice_pct,
            },
            "szczegoly_po": {
                "sr_dlug_zdania": mod.avg_sentence_length,
                "odch_dlug_zdania": mod.sentence_length_std,
                "unikat_slow": mod.unique_word_ratio,
                "formalne_wzorce": len(mod.formal_patterns_found),
                "strona_bierna": mod.passive_voice_pct,
            },
        }

    # ---- sentence splitting -----------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        """Podziel tekst na zdania z uwzglednieniem polskich skrotow."""
        # Krok 1: zamien skroty na placeholder zeby kropki nie dzielily zdan
        protected = text
        placeholder_map: Dict[str, str] = {}

        for abbr in sorted(_PL_ABBREVIATIONS, key=len, reverse=True):
            pattern = re.compile(
                r"\b(" + re.escape(abbr) + r")\.\s*",
                re.IGNORECASE,
            )
            for m in pattern.finditer(protected):
                original = m.group(0)
                key = f"\x00ABBR{len(placeholder_map):04d}\x00 "
                placeholder_map[key.strip()] = original
                protected = protected.replace(original, key, 1)

        # Krok 2: ochron liczby z kropka (np. "3.14", "rys. 5" juz pokryte)
        num_pattern = re.compile(r"(\d+)\.(\d+)")
        for m in num_pattern.finditer(protected):
            original = m.group(0)
            key = f"\x00NUM{len(placeholder_map):04d}\x00"
            placeholder_map[key] = original
            protected = protected.replace(original, key, 1)

        # Krok 3: split na .!? po ktorych jest spacja + wielka litera lub koniec
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\u0104\u0106\u0118\u0141\u0143\u00d3\u015a\u0179\u017b])", protected)

        # Krok 4: przywroc placeholdery
        sentences = []
        for part in parts:
            s = part.strip()
            for key, orig in placeholder_map.items():
                s = s.replace(key, orig)
            if len(s) >= MIN_SENTENCE_LENGTH:
                sentences.append(s)

        return sentences

    # ---- per-sentence analysis -------------------------------------------

    def _analyze_sentence(self, sentence: str) -> SentenceAnalysis:
        """Analizuj pojedyncze zdanie pod katem wzorcow AI."""
        words = sentence.split()
        wc = len(words)
        issues: List[str] = []
        suggestions: List[str] = []
        risk = 0.0

        # 1. Dlugosc zdania
        if wc > MAX_GOOD_SENTENCE_WORDS:
            issues.append(f"Za dlugie zdanie ({wc} slow)")
            target = math.ceil(wc / 2)
            suggestions.append(
                f"To zdanie ma {wc} slow - podziel na 2 krotsze "
                f"(ok. {target} slow kazde)"
            )
            risk += min(0.3, (wc - MAX_GOOD_SENTENCE_WORDS) * 0.01)

        # 2. Formalne poczatki
        for pat in _FORMAL_STARTS:
            if pat.search(sentence):
                matched = pat.pattern.lstrip("^").replace("\\b", "").replace("\\s+", " ")
                issues.append(f"Formalny poczatek zdania")
                suggestions.append(
                    f"Zamien formalny poczatek na bardziej naturalny"
                )
                risk += 0.2
                break

        # 3. Strona bierna
        passive_count = sum(1 for p in _PASSIVE_RE if p.search(sentence))
        if passive_count > 0:
            issues.append(f"Strona bierna ({passive_count}x)")
            suggestions.append("Uzyj strony czynnej zamiast biernej")
            risk += 0.15 * passive_count

        # 4. Hedging
        for pat, name in _HEDGING_PATTERNS:
            if pat.search(sentence):
                issues.append(f"Ostroznosciowe sformulowanie: '{name}'")
                # Szukaj zamiennika
                for label, _, alts in REPLACEMENT_TABLE:
                    if name in label or label in name:
                        suggestions.append(
                            f"Zamien '{name}' na '{alts[0]}'"
                        )
                        break
                else:
                    suggestions.append(
                        f"Uprosc sformulowanie '{name}'"
                    )
                risk += 0.15

        # 5. Sprawdz konkretne frazy z tabeli zamiennikow
        for label, regex, alts in REPLACEMENT_TABLE:
            if re.search(regex, sentence, re.IGNORECASE):
                if not any(label in i for i in issues):
                    issues.append(f"Formulaiczna fraza: '{label}'")
                    alt_str = "' lub '".join(alts)
                    suggestions.append(
                        f"Zamien '{label}' na '{alt_str}'"
                    )
                    risk += 0.15

        risk = min(risk, 1.0)

        return SentenceAnalysis(
            text=sentence,
            word_count=wc,
            issues=issues,
            suggestions=suggestions,
            risk_score=round(risk, 2),
        )

    # ---- text-level checks -----------------------------------------------

    def _check_formal_patterns(self, text: str) -> List[str]:
        """Znajdz formalne wzorce z konfiguracji."""
        found = []
        for pattern, description in FORMAL_PATTERNS_PL:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(description)
        return found

    def _check_repetitive_starts(self, sentences: List[str]) -> List[tuple]:
        """Sprawdz czy zdania zaczynaja sie od tych samych slow."""
        first_words = []
        for s in sentences:
            words = s.split()
            if words:
                first_words.append(words[0].lower().rstrip(",.;:"))

        counts = Counter(first_words)
        return [
            (word, count)
            for word, count in counts.most_common()
            if count >= 3
        ]

    def _calculate_passive_voice_pct(self, sentences: List[str]) -> float:
        """Oblicz procent zdan ze strona bierna."""
        if not sentences:
            return 0.0
        passive_count = 0
        for s in sentences:
            if any(p.search(s) for p in _PASSIVE_RE):
                passive_count += 1
        return (passive_count / len(sentences)) * 100

    # ---- risk scoring ----------------------------------------------------

    def _overall_risk(self, report: StyleReport) -> float:
        """Oblicz sumaryczne ryzyko AI (0-100)."""
        scores: Dict[str, float] = {}

        # 1. Wariancja dlugosci zdan (30%) - niska wariancja = AI
        if report.sentence_length_std < MIN_GOOD_VARIANCE:
            # Im mniejsza wariancja, tym wyzsze ryzyko
            variance_score = 1.0 - (report.sentence_length_std / MIN_GOOD_VARIANCE)
        else:
            variance_score = 0.0
        scores["wariancja_zdan"] = variance_score * 30

        # 2. Formalne wzorce (25%)
        n_formal = len(report.formal_patterns_found)
        formal_score = min(n_formal / 5.0, 1.0)
        scores["formalne_wzorce"] = formal_score * 25

        # 3. Unikatowe slowa (20%) - niski stosunek = formulaiczny
        if report.unique_word_ratio < MIN_GOOD_UNIQUE_RATIO:
            unique_score = 1.0 - (report.unique_word_ratio / MIN_GOOD_UNIQUE_RATIO)
        else:
            unique_score = max(0.0, 1.0 - report.unique_word_ratio)
            unique_score = min(unique_score, 0.3)
        scores["unikat_slow"] = unique_score * 20

        # 4. Strona bierna (15%)
        passive_score = min(report.passive_voice_pct / 50.0, 1.0)
        scores["strona_bierna"] = passive_score * 15

        # 5. Powtarzalne poczatki (10%)
        if report.repetitive_starts:
            max_rep = max(c for _, c in report.repetitive_starts)
            total_sentences = len(report.sentences) if report.sentences else 1
            rep_score = min(max_rep / total_sentences * 2, 1.0)
        else:
            rep_score = 0.0
        scores["powtorzenia"] = rep_score * 10

        total = sum(scores.values())
        return round(min(total, 100.0), 1)

    # ---- suggestions -----------------------------------------------------

    def _global_suggestions(self, report: StyleReport) -> List[str]:
        """Globalne sugestie poprawy na podstawie raportu."""
        sugs: List[str] = []

        if report.sentence_length_std < MIN_GOOD_VARIANCE:
            sugs.append(
                f"Zdania maja zbyt regularna dlugosc (odch. std = {report.sentence_length_std}). "
                f"Mieszaj krotkie zdania (8-12 slow) z dluzszymi (18-25 slow)."
            )

        if report.unique_word_ratio < MIN_GOOD_UNIQUE_RATIO:
            sugs.append(
                f"Niski wskaznik unikalnych slow ({report.unique_word_ratio:.1%}). "
                f"Uzywaj synonimow i roznorodnego slownictwa."
            )

        if report.formal_patterns_found:
            sugs.append(
                f"Znaleziono {len(report.formal_patterns_found)} formalnych wzorcow. "
                f"Zamien je na bardziej naturalne sformulowania (patrz tabela zamiennikow)."
            )

        if report.repetitive_starts:
            words_str = ", ".join(f"'{w}' ({c}x)" for w, c in report.repetitive_starts)
            sugs.append(
                f"Powtarzalne poczatki zdan: {words_str}. "
                f"Zroznicuj poczatki zdan."
            )

        if report.passive_voice_pct > 30:
            sugs.append(
                f"Wysoki udzial strony biernej ({report.passive_voice_pct:.0f}%). "
                f"Zamien na strone czynna tam, gdzie to mozliwe."
            )

        if report.avg_sentence_length > MAX_GOOD_SENTENCE_WORDS:
            sugs.append(
                f"Srednia dlugosc zdania ({report.avg_sentence_length:.0f} slow) jest za wysoka. "
                f"Cel: ponizej {MAX_GOOD_SENTENCE_WORDS} slow."
            )

        return sugs

    # ---- report generation ------------------------------------------------

    def generate_report(self, analyses: List[StyleReport]) -> str:
        """Generuj raport markdown z wynikow analizy."""
        lines: List[str] = []
        lines.append("# Raport analizy stylu - wykrywanie wzorcow AI\n")
        lines.append(f"Przeanalizowano {len(analyses)} fragmentow tekstu.\n")

        # ---- Tabela podsumowania ----
        lines.append("## Podsumowanie\n")
        lines.append("| Chunk | Ryzyko AI | Klasa | Sr. dlug. zdania | Odch. std | Unikat slow | Str. bierna | Formalne |")
        lines.append("|-------|-----------|-------|------------------|-----------|-------------|-------------|----------|")

        for r in analyses:
            emoji = THRESHOLDS.emoji(r.overall_risk)
            label = THRESHOLDS.label_pl(r.overall_risk)
            lines.append(
                f"| {r.chunk_id:02d} "
                f"| {r.overall_risk:.0f}% "
                f"| {emoji} {label} "
                f"| {r.avg_sentence_length:.1f} "
                f"| {r.sentence_length_std:.1f} "
                f"| {r.unique_word_ratio:.1%} "
                f"| {r.passive_voice_pct:.0f}% "
                f"| {len(r.formal_patterns_found)} |"
            )

        # ---- Srednie ----
        if analyses:
            avg_risk = statistics.mean(r.overall_risk for r in analyses)
            lines.append(f"\n**Srednie ryzyko AI:** {avg_risk:.1f}%\n")

        # ---- Szczegoly chunkow z wysokim ryzykiem ----
        high_risk = [r for r in analyses if r.overall_risk >= THRESHOLDS.safe]
        if high_risk:
            lines.append("## Fragmenty wymagajace uwagi\n")
            for r in sorted(high_risk, key=lambda x: -x.overall_risk):
                lines.append(f"### Chunk {r.chunk_id:02d} - ryzyko {r.overall_risk:.0f}%\n")

                if r.suggestions:
                    lines.append("**Globalne sugestie:**\n")
                    for s in r.suggestions:
                        lines.append(f"- {s}")
                    lines.append("")

                if r.formal_patterns_found:
                    lines.append("**Znalezione formalne wzorce:**\n")
                    for p in r.formal_patterns_found:
                        lines.append(f"- {p}")
                    lines.append("")

                if r.repetitive_starts:
                    lines.append("**Powtarzalne poczatki zdan:**\n")
                    for word, count in r.repetitive_starts:
                        lines.append(f"- '{word}' - {count} razy")
                    lines.append("")

                # Zdania z problemami
                problem_sentences = [
                    sa for sa in r.sentences if sa.issues
                ]
                if problem_sentences:
                    lines.append("**Problematyczne zdania:**\n")
                    for i, sa in enumerate(problem_sentences[:10], 1):
                        preview = sa.text[:120] + ("..." if len(sa.text) > 120 else "")
                        lines.append(f"{i}. `{preview}`")
                        lines.append(f"   - Ryzyko: {sa.risk_score:.0%}")
                        for iss in sa.issues:
                            lines.append(f"   - Problem: {iss}")
                        for sug in sa.suggestions:
                            lines.append(f"   - Sugestia: {sug}")
                        lines.append("")

        # ---- Tabela zamiennikow ----
        lines.append("## Tabela zamiennikow fraz\n")
        lines.append("| Fraza oryginalna | Zamiennik(i) |")
        lines.append("|-----------------|--------------|")
        for label, _, alts in REPLACEMENT_TABLE:
            alt_str = " / ".join(alts)
            lines.append(f"| {label} | {alt_str} |")

        lines.append("")
        lines.append("---\n")
        lines.append("*Raport wygenerowany przez style_analyzer.py*\n")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    analyzer = StyleAnalyzer()

    # Pojedynczy plik z argumentu CLI
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
        if not filepath.exists():
            log.error(f"Plik nie istnieje: {filepath}")
            sys.exit(1)

        text = filepath.read_text(encoding="utf-8")
        # Probuj wyciagnac chunk_id z nazwy pliku
        try:
            chunk_id = int(filepath.stem.split("_")[1])
        except (ValueError, IndexError):
            chunk_id = 0

        report = analyzer.analyze_text(text, chunk_id)
        _print_single_report(report)

        md = analyzer.generate_report([report])
        save_markdown_report(md, f"styl_chunk_{chunk_id:02d}.md")
        return

    # Wszystkie chunki
    chunks = load_chunks()
    if not chunks:
        log.error("Brak chunkow do analizy. Sprawdz katalog chunks/.")
        sys.exit(1)

    log.info(f"Analizuje {len(chunks)} chunkow...")
    analyses: List[StyleReport] = []

    for chunk_id, text in sorted(chunks.items()):
        report = analyzer.analyze_text(text, chunk_id)
        analyses.append(report)
        _print_chunk_summary(report)

    # Podsumowanie
    print("\n" + "=" * 70)
    print("PODSUMOWANIE ANALIZY STYLU")
    print("=" * 70)

    avg_risk = statistics.mean(r.overall_risk for r in analyses)
    max_risk = max(r.overall_risk for r in analyses)
    min_risk = min(r.overall_risk for r in analyses)
    high_count = sum(1 for r in analyses if r.overall_risk >= THRESHOLDS.safe)

    print(f"  Przeanalizowanych fragmentow: {len(analyses)}")
    print(f"  Srednie ryzyko AI:            {avg_risk:.1f}%")
    print(f"  Najnizsze ryzyko:             {min_risk:.1f}%")
    print(f"  Najwyzsze ryzyko:             {max_risk:.1f}%")
    print(f"  Fragmenty wymagajace uwagi:   {high_count}/{len(analyses)}")
    print()

    # Najgorsze chunki
    worst = sorted(analyses, key=lambda r: -r.overall_risk)[:5]
    print("Top 5 chunkow z najwyzszym ryzykiem:")
    for r in worst:
        emoji = THRESHOLDS.emoji(r.overall_risk)
        label = THRESHOLDS.label_pl(r.overall_risk)
        print(f"  Chunk {r.chunk_id:02d}: {r.overall_risk:.0f}% {emoji} {label}")

    # Zapisz raport
    md = analyzer.generate_report(analyses)
    report_path = save_markdown_report(md, "RAPORT_STYL.md")
    print(f"\nRaport zapisany: {report_path}")


def _print_single_report(report: StyleReport):
    """Wyswietl szczegolowy raport dla jednego fragmentu."""
    emoji = THRESHOLDS.emoji(report.overall_risk)
    label = THRESHOLDS.label_pl(report.overall_risk)

    print(f"\n{'=' * 70}")
    print(f"ANALIZA STYLU - Chunk {report.chunk_id:02d}")
    print(f"{'=' * 70}")
    print(f"  Ryzyko AI:              {report.overall_risk:.0f}% {emoji} {label}")
    print(f"  Sr. dlugosc zdania:     {report.avg_sentence_length:.1f} slow")
    print(f"  Odch. std dlugosci:     {report.sentence_length_std:.1f}")
    print(f"  Wskaznik unikat. slow:  {report.unique_word_ratio:.1%}")
    print(f"  Strona bierna:          {report.passive_voice_pct:.0f}%")
    print(f"  Formalne wzorce:        {len(report.formal_patterns_found)}")
    print()

    if report.suggestions:
        print("Sugestie poprawy:")
        for s in report.suggestions:
            print(f"  - {s}")
        print()

    if report.formal_patterns_found:
        print("Znalezione formalne wzorce:")
        for p in report.formal_patterns_found:
            print(f"  - {p}")
        print()

    problem_sentences = [sa for sa in report.sentences if sa.issues]
    if problem_sentences:
        print(f"Problematyczne zdania ({len(problem_sentences)}/{len(report.sentences)}):")
        for i, sa in enumerate(problem_sentences[:15], 1):
            preview = sa.text[:100] + ("..." if len(sa.text) > 100 else "")
            print(f"\n  {i}. [{sa.risk_score:.0%}] {preview}")
            for iss in sa.issues:
                print(f"     Problem: {iss}")
            for sug in sa.suggestions:
                print(f"     Sugestia: {sug}")


def _print_chunk_summary(report: StyleReport):
    """Wyswietl krotkie podsumowanie jednego chunku."""
    emoji = THRESHOLDS.emoji(report.overall_risk)
    n_issues = sum(1 for sa in report.sentences if sa.issues)
    print(
        f"  Chunk {report.chunk_id:02d}: "
        f"ryzyko {report.overall_risk:5.1f}% {emoji}  "
        f"zdania: {len(report.sentences):2d}  "
        f"problemy: {n_issues:2d}  "
        f"formalne: {len(report.formal_patterns_found):2d}  "
        f"bierna: {report.passive_voice_pct:4.0f}%"
    )


if __name__ == "__main__":
    main()
