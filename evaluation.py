"""
Kompleksowy framework ewaluacji detektorow AI-generated text.
Zastepuje data_science_validation.py -- kluczowe usprawnienia:

1. Polskie probki AI (nie angielskie!) -- uczciwe porownanie z polskim tekstem ludzkim
2. Wiecej metryk: AUROC, TPR@FPR=1%, Cohen's d, bootstrap CI
3. Lepsza wizualizacja: confusion matrix, ROC data, raport markdown

Uzycie: python evaluation.py
"""

import math
import random
import time
import logging
from typing import List, Dict, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field, asdict

from config import THRESHOLDS, RESULTS_DIR
from utils import (
    DetectionResult,
    load_chunks,
    call_zerogpt,
    call_contentdetector,
    setup_logging,
    save_json_report,
    save_markdown_report,
)


log = setup_logging("evaluation")


# ============================================================================
# GENERYCZNE POLSKIE PROBKI AI-GENERATED
# Celowo napisane "AI-like": formalne, jednolite zdania, szablonowe frazy.
# Tematyka ogolnoakademicka - niezalezna od konkretnej pracy.
# ============================================================================

POLISH_AI_SAMPLES: Dict[str, str] = {
    "ai_przeglad_literatury": (
        "Przeglad literatury przedmiotu stanowi fundamentalny element kazdej pracy "
        "badawczej. Nalezy podkreslic, ze analiza istniejacych badan pozwala na "
        "identyfikacje luk w dotychczasowej wiedzy oraz na sformulowanie precyzyjnych "
        "pytan badawczych. W niniejszej pracy dokonano systematycznego przegladu "
        "publikacji z ostatnich pieciu lat, wykorzystujac bazy danych Scopus, Web of "
        "Science oraz Google Scholar. W zwiazku z powyzszym, istotnym aspektem jest "
        "okreslenie kryteriow wlaczenia i wylaczenia poszczegolnych pozycji "
        "bibliograficznych. Na podstawie przeprowadzonej analizy mozna stwierdzic, "
        "ze obserwuje sie znaczacy wzrost zainteresowania omawiana problematyka "
        "w literaturze naukowej. Zagadnienie to wymaga dalszych badan empirycznych "
        "w celu weryfikacji dotychczasowych ustalen teoretycznych."
    ),

    "ai_metodologia_badawcza": (
        "Metodologia badawcza zastosowana w niniejszej pracy opiera sie na podejsciu "
        "mieszanym, laczacym metody ilosciowe z jakosciowymi. Nalezy podkreslic, ze "
        "dobor metod badawczych zostal podyktowany specyfika analizowanego problemu "
        "oraz charakterem postawionych pytan badawczych. Zbior danych zostal "
        "przygotowany z zachowaniem rygorystycznych standardow jakosciowych, "
        "obejmujacych walidacje krizowa oraz stratyfikowany podzial na zbiory "
        "treningowe i testowe. Istotnym aspektem jest rowniez zapewnienie "
        "powtarzalnosci eksperymentow poprzez udostepnienie kodu zrodlowego "
        "oraz szczegolowej dokumentacji procedur badawczych. Podsumowujac, mozna "
        "stwierdzic, ze przyjeta metodologia spelnia wspolczesne standardy "
        "prowadzenia badan naukowych w danej dziedzinie."
    ),

    "ai_wyniki_eksperymentalne": (
        "Ewaluacja eksperymentalna zostala przeprowadzona z wykorzystaniem "
        "standaryzowanych protokolow benchmarkowych w celu zapewnienia "
        "odtwarzalnosci i uczciwego porownania miedzy analizowanymi podejsciami. "
        "Konfiguracja eksperymentalna wykorzystala optymalizator AdamW z kosinusowym "
        "harmonogramem szybkosci uczenia oraz liniowym rozgrzewaniem w poczatkowych "
        "epokach. Trening z mieszana precyzja za posrednictwem frameworka AMP "
        "biblioteki PyTorch umozliwil efektywne wykorzystanie zasobow obliczeniowych "
        "przy jednoczesnym zachowaniu stabilnosci numerycznej. Wczesne zatrzymanie z "
        "parametrem cierpliwosci zapobieglo przeuczeniu poprzez monitorowanie "
        "metryk walidacyjnych. Framework benchmarkowy automatycznie generuje "
        "kompleksowe raporty ewaluacyjne obejmujace dokladnosc, AUC-ROC, "
        "F1-score oraz macierze pomylek. Podsumowujac, mozna stwierdzic, ze "
        "zaproponowana konfiguracja eksperymentalna spelnia wspolczesne standardy "
        "metodologiczne w dziedzinie badan naukowych."
    ),

    "ai_analiza_wynikow": (
        "Analiza uzyskanych wynikow pozwala na sformulowanie kilku istotnych "
        "wnioskow dotyczacych skutecznosci zaproponowanego podejscia. W kontekscie "
        "przeprowadzonych eksperymentow, zaproponowana metoda osiagnela wyniki "
        "przewyzszajace dotychczasowe rozwiazania referencyjne. Nalezy podkreslic, "
        "ze poprawa wyniku jest statystycznie istotna przy poziomie ufnosci 95%. "
        "Badania wykazuja, ze kluczowym czynnikiem wplywajacym na skutecznosc "
        "jest odpowiedni dobor hiperparametrow oraz architektura modelu. Istotnym "
        "aspektem jest rowniez analiza przypadkow, w ktorych model popelnia bledy, "
        "co pozwala na identyfikacje potencjalnych kierunkow dalszych usprawnien. "
        "Na podstawie przeprowadzonych badan mozna stwierdzic, ze zaproponowane "
        "podejscie stanowi wartosciowy wklad w rozwoj omawianej dziedziny."
    ),

    "ai_podsumowanie_wnioski": (
        "Podsumowujac, w niniejszej pracy przedstawiono kompleksowe podejscie "
        "do omawianego zagadnienia badawczego. Majac na uwadze uzyskane wyniki, "
        "mozna stwierdzic, ze cele pracy zostaly w pelni zrealizowane. Glownym "
        "wkladem niniejszej pracy jest opracowanie nowej metody, ktora wykazuje "
        "przewage nad dotychczasowymi rozwiazaniami pod wzgledem skutecznosci "
        "oraz efektywnosci obliczeniowej. W zwiazku z powyzszym, zaproponowane "
        "podejscie moze znalezc zastosowanie zarowno w kontekscie akademickim, "
        "jak i w praktycznych wdrozeniach przemyslowych. Powyzsze rozwazania "
        "potwierdzaja zasadnosc kontynuowania badan w omawianym kierunku. "
        "Dalsze prace powinny koncentrowac sie na rozszerzeniu zakresu "
        "eksperymentow oraz na weryfikacji skalowalnosci zaproponowanego rozwiazania."
    ),
}


# ============================================================================
# EVALUATION SUITE
# ============================================================================

@dataclass
class ConfusionMatrix:
    tp: int = 0  # AI correctly detected as AI
    fp: int = 0  # Human incorrectly flagged as AI
    tn: int = 0  # Human correctly identified as human
    fn: int = 0  # AI incorrectly classified as human

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class DetectorEvaluation:
    """Wyniki ewaluacji jednego detektora."""
    detector_name: str
    confusion: ConfusionMatrix
    auroc: float
    tpr_at_fpr_1pct: float
    cohens_d: float
    avg_ai_score_on_human: float
    avg_ai_score_on_ai: float
    discrimination_gap: float
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    roc_points: List[Tuple[float, float]]  # (FPR, TPR) for ROC curve
    all_results: List[Dict]


class EvaluationSuite:
    """
    Framework ewaluacji detektorow AI-generated text.
    Porownuje detektory na zbiorze ludzkim i AI-generowanym (oba po polsku).
    """

    CLASSIFICATION_THRESHOLD = 50.0  # >= 50% AI => classified as AI

    def __init__(self, detectors: List[Callable]):
        """
        Args:
            detectors: lista funkcji detektorow, kazda przyjmuje (text: str) -> float
                       zwraca AI% (0-100) lub -1 przy bledzie.
        """
        self.detectors = detectors
        self.evaluations: Dict[str, DetectorEvaluation] = {}

    def run(
        self,
        human_chunks: Dict[int, str],
        ai_samples: Dict[str, str],
        delay_between_calls: float = 3.0,
    ) -> Dict[str, DetectorEvaluation]:
        """
        Uruchom pelna ewaluacje.

        Args:
            human_chunks: {chunk_id: text} -- oryginalne ludzkie teksty
            ai_samples: {sample_id: text} -- AI-generated po polsku
            delay_between_calls: opoznienie miedzy wywolaniami API (sekundy)

        Returns:
            Dict z wynikami per detektor
        """
        for detector_fn in self.detectors:
            detector_name = detector_fn.__name__
            log.info(f"=== Ewaluacja detektora: {detector_name} ===")

            y_true: List[int] = []       # 1 = AI, 0 = human
            y_scores: List[float] = []   # raw AI% score
            all_results: List[Dict] = []

            human_scores: List[float] = []
            ai_scores: List[float] = []

            # --- Test na ludzkich tekstach ---
            log.info(f"Testowanie {len(human_chunks)} chunkow ludzkich...")
            for chunk_id, text in sorted(human_chunks.items()):
                score = detector_fn(text)
                if score < 0:
                    log.warning(
                        f"  {detector_name} chunk_{chunk_id:02d}: blad API, pomijam"
                    )
                    continue

                y_true.append(0)
                y_scores.append(score)
                human_scores.append(score)

                predicted_ai = score >= self.CLASSIFICATION_THRESHOLD
                all_results.append({
                    "sample_id": f"human_chunk_{chunk_id:02d}",
                    "is_ai": False,
                    "ai_score": round(score, 2),
                    "predicted_ai": predicted_ai,
                    "correct": not predicted_ai,
                })

                log.info(
                    f"  chunk_{chunk_id:02d} [HUMAN]: {score:.1f}% AI "
                    f"-> {'FP!' if predicted_ai else 'OK'}"
                )
                time.sleep(delay_between_calls)

            # --- Test na AI-generated ---
            log.info(f"Testowanie {len(ai_samples)} probek AI...")
            for sample_id, text in ai_samples.items():
                score = detector_fn(text)
                if score < 0:
                    log.warning(
                        f"  {detector_name} {sample_id}: blad API, pomijam"
                    )
                    continue

                y_true.append(1)
                y_scores.append(score)
                ai_scores.append(score)

                predicted_ai = score >= self.CLASSIFICATION_THRESHOLD
                all_results.append({
                    "sample_id": sample_id,
                    "is_ai": True,
                    "ai_score": round(score, 2),
                    "predicted_ai": predicted_ai,
                    "correct": predicted_ai,
                })

                log.info(
                    f"  {sample_id} [AI]: {score:.1f}% AI "
                    f"-> {'OK' if predicted_ai else 'FN!'}"
                )
                time.sleep(delay_between_calls)

            # --- Oblicz metryki ---
            if len(y_true) < 2:
                log.error(f"{detector_name}: za malo wynikow ({len(y_true)}), pomijam")
                continue

            metrics = self._calculate_metrics(
                y_true, y_scores, human_scores, ai_scores, all_results
            )
            self.evaluations[detector_name] = metrics
            self.evaluations[detector_name].detector_name = detector_name

            log.info(
                f"  {detector_name} AUROC={metrics.auroc:.3f} "
                f"Accuracy={metrics.confusion.accuracy:.1%} "
                f"Cohen's d={metrics.cohens_d:.2f}"
            )

        return self.evaluations

    def _calculate_metrics(
        self,
        y_true: List[int],
        y_scores: List[float],
        human_scores: List[float],
        ai_scores: List[float],
        all_results: List[Dict],
    ) -> DetectorEvaluation:
        """Oblicz wszystkie metryki dla jednego detektora."""

        # Confusion matrix
        cm = ConfusionMatrix()
        for label, score in zip(y_true, y_scores):
            predicted_ai = score >= self.CLASSIFICATION_THRESHOLD
            if label == 1 and predicted_ai:
                cm.tp += 1
            elif label == 0 and predicted_ai:
                cm.fp += 1
            elif label == 0 and not predicted_ai:
                cm.tn += 1
            else:
                cm.fn += 1

        # AUROC
        auroc, roc_points = self._calculate_auroc(y_true, y_scores)

        # TPR @ FPR=1%
        tpr_at_1pct = self._calculate_tpr_at_fpr(y_true, y_scores, target_fpr=0.01)

        # Cohen's d
        cohens_d = self._calculate_cohens_d(ai_scores, human_scores)

        # Average scores
        avg_human = sum(human_scores) / len(human_scores) if human_scores else 0.0
        avg_ai = sum(ai_scores) / len(ai_scores) if ai_scores else 0.0
        gap = avg_ai - avg_human

        # Bootstrap CI for accuracy
        ci_lower, ci_upper = self._bootstrap_ci(y_true, y_scores)

        return DetectorEvaluation(
            detector_name="",
            confusion=cm,
            auroc=auroc,
            tpr_at_fpr_1pct=tpr_at_1pct,
            cohens_d=cohens_d,
            avg_ai_score_on_human=round(avg_human, 2),
            avg_ai_score_on_ai=round(avg_ai, 2),
            discrimination_gap=round(gap, 2),
            accuracy_ci_lower=round(ci_lower, 4),
            accuracy_ci_upper=round(ci_upper, 4),
            roc_points=roc_points,
            all_results=all_results,
        )

    @staticmethod
    def _calculate_auroc(
        y_true: List[int], y_scores: List[float]
    ) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Reczna implementacja AUROC (bez sklearn).
        Sortuje probki malejaco po score, oblicza TPR/FPR przy kazdym progu,
        calkuje metoda trapezow.

        Returns:
            (auroc, roc_points) -- roc_points to lista (FPR, TPR)
        """
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5, [(0.0, 0.0), (1.0, 1.0)]

        # Polacz i sortuj malejaco po score
        combined = sorted(zip(y_scores, y_true), key=lambda x: -x[0])

        roc_points: List[Tuple[float, float]] = [(0.0, 0.0)]
        tp_count = 0
        fp_count = 0

        prev_score = None
        for score, label in combined:
            # Przy zmianie progu zapisz punkt ROC
            if prev_score is not None and score != prev_score:
                fpr = fp_count / n_neg
                tpr = tp_count / n_pos
                roc_points.append((fpr, tpr))

            if label == 1:
                tp_count += 1
            else:
                fp_count += 1
            prev_score = score

        # Ostatni punkt
        roc_points.append((fp_count / n_neg, tp_count / n_pos))

        # Upewnij sie, ze koncowy punkt to (1.0, 1.0)
        if roc_points[-1] != (1.0, 1.0):
            roc_points.append((1.0, 1.0))

        # Calkowanie metoda trapezow
        auroc = 0.0
        for i in range(1, len(roc_points)):
            x0, y0 = roc_points[i - 1]
            x1, y1 = roc_points[i]
            auroc += (x1 - x0) * (y0 + y1) / 2.0

        return round(auroc, 4), roc_points

    @staticmethod
    def _calculate_tpr_at_fpr(
        y_true: List[int], y_scores: List[float], target_fpr: float = 0.01
    ) -> float:
        """
        Oblicz TPR przy zadanym FPR (domyslnie 1%).
        Kluczowe w kontekscie akademickim -- falszywe oskarzenia sa kosztowne.
        """
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.0

        combined = sorted(zip(y_scores, y_true), key=lambda x: -x[0])

        best_tpr = 0.0
        tp_count = 0
        fp_count = 0

        for score, label in combined:
            if label == 1:
                tp_count += 1
            else:
                fp_count += 1

            fpr = fp_count / n_neg
            tpr = tp_count / n_pos

            if fpr <= target_fpr:
                best_tpr = tpr
            else:
                break

        return round(best_tpr, 4)

    @staticmethod
    def _calculate_cohens_d(
        group1_scores: List[float], group2_scores: List[float]
    ) -> float:
        """
        Cohen's d -- wielkosc efektu miedzy dwoma grupami.
        group1 = AI scores, group2 = human scores.
        d > 0.8 = duzy efekt (dobra separacja).
        """
        if len(group1_scores) < 2 or len(group2_scores) < 2:
            return 0.0

        mean1 = sum(group1_scores) / len(group1_scores)
        mean2 = sum(group2_scores) / len(group2_scores)

        var1 = sum((x - mean1) ** 2 for x in group1_scores) / (len(group1_scores) - 1)
        var2 = sum((x - mean2) ** 2 for x in group2_scores) / (len(group2_scores) - 1)

        # Pooled standard deviation
        n1, n2 = len(group1_scores), len(group2_scores)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else 1e-10

        d = (mean1 - mean2) / pooled_sd
        return round(d, 4)

    @staticmethod
    def _bootstrap_ci(
        y_true: List[int],
        y_scores: List[float],
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        threshold: float = 50.0,
    ) -> Tuple[float, float]:
        """
        Bootstrap 95% CI dla accuracy.
        Losuje z powtorzeniami i liczy accuracy na kazdej probce.
        """
        n = len(y_true)
        if n == 0:
            return (0.0, 0.0)

        rng = random.Random(42)  # reproducible
        accuracies: List[float] = []

        for _ in range(n_bootstrap):
            indices = [rng.randint(0, n - 1) for _ in range(n)]
            correct = 0
            for idx in indices:
                predicted_ai = y_scores[idx] >= threshold
                actual_ai = y_true[idx] == 1
                if predicted_ai == actual_ai:
                    correct += 1
            accuracies.append(correct / n)

        accuracies.sort()
        alpha = 1 - ci
        lower_idx = int(math.floor(alpha / 2 * n_bootstrap))
        upper_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
        lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
        upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

        return (accuracies[lower_idx], accuracies[upper_idx])

    def generate_report(self) -> str:
        """Generuj raport markdown z wynikami ewaluacji."""
        if not self.evaluations:
            return "# Brak wynikow ewaluacji\n\nNie przeprowadzono zadnych testow."

        lines: List[str] = []
        lines.append("# Raport ewaluacji detektorow AI-generated text")
        lines.append("")
        lines.append("## Metodologia")
        lines.append("")
        lines.append(
            "Ewaluacja porownuje detektory na dwoch zbiorach tekstow **w jezyku polskim**:"
        )
        lines.append(
            "- **Zbiory ludzkie**: oryginalne fragmenty pracy akademickiej"
        )
        lines.append(
            "- **Zbiory AI**: polskie teksty akademickie wygenerowane w stylu AI"
        )
        lines.append("")
        lines.append(
            "Prog klasyfikacji: >= 50% AI => zakwalifikowane jako AI-generated."
        )
        lines.append("")

        # --- Tabela podsumowujaca ---
        lines.append("## Podsumowanie")
        lines.append("")
        lines.append(
            "| Detektor | Accuracy | AUROC | Precision | Recall | F1 | "
            "Cohen's d | TPR@FPR=1% | 95% CI Accuracy |"
        )
        lines.append(
            "|----------|----------|-------|-----------|--------|----|"
            "-----------|------------|-----------------|"
        )

        for name, ev in self.evaluations.items():
            cm = ev.confusion
            lines.append(
                f"| {name} "
                f"| {cm.accuracy:.1%} "
                f"| {ev.auroc:.3f} "
                f"| {cm.precision:.1%} "
                f"| {cm.recall:.1%} "
                f"| {cm.f1:.3f} "
                f"| {ev.cohens_d:.2f} "
                f"| {ev.tpr_at_fpr_1pct:.1%} "
                f"| [{ev.accuracy_ci_lower:.1%}, {ev.accuracy_ci_upper:.1%}] |"
            )

        lines.append("")

        # --- Szczegoly per detektor ---
        for name, ev in self.evaluations.items():
            cm = ev.confusion
            lines.append(f"## Detektor: {name}")
            lines.append("")

            # Metryki
            lines.append("### Metryki")
            lines.append("")
            lines.append(f"- **Accuracy**: {cm.accuracy:.1%}")
            lines.append(f"- **Precision**: {cm.precision:.1%}")
            lines.append(f"- **Recall**: {cm.recall:.1%}")
            lines.append(f"- **F1-score**: {cm.f1:.3f}")
            lines.append(f"- **AUROC**: {ev.auroc:.4f}")
            lines.append(f"- **TPR @ FPR=1%**: {ev.tpr_at_fpr_1pct:.1%}")
            lines.append(f"- **Cohen's d**: {ev.cohens_d:.2f}")
            lines.append(
                f"- **95% Bootstrap CI (accuracy)**: "
                f"[{ev.accuracy_ci_lower:.1%}, {ev.accuracy_ci_upper:.1%}]"
            )
            lines.append("")

            # Srednie wyniki
            lines.append("### Srednie wyniki")
            lines.append("")
            lines.append(
                f"- Sredni AI% na tekstach **ludzkich**: "
                f"{ev.avg_ai_score_on_human:.1f}% (powinno byc niskie)"
            )
            lines.append(
                f"- Sredni AI% na tekstach **AI**: "
                f"{ev.avg_ai_score_on_ai:.1f}% (powinno byc wysokie)"
            )
            lines.append(f"- **Luka dyskryminacyjna**: {ev.discrimination_gap:.1f} pp")
            lines.append("")

            # Confusion matrix
            lines.append("### Macierz pomylek")
            lines.append("")
            lines.append("|  | Predicted Human | Predicted AI |")
            lines.append("|--|-----------------|--------------|")
            lines.append(
                f"| **Actual Human** | TN = {cm.tn} | FP = {cm.fp} |"
            )
            lines.append(
                f"| **Actual AI** | FN = {cm.fn} | TP = {cm.tp} |"
            )
            lines.append("")

            # Szczegolowe wyniki
            lines.append("### Szczegolowe wyniki per probka")
            lines.append("")
            lines.append("| Probka | Rzeczywista | AI% | Predykcja | Poprawna |")
            lines.append("|--------|-------------|-----|-----------|----------|")
            for r in ev.all_results:
                actual = "AI" if r["is_ai"] else "Human"
                pred = "AI" if r["predicted_ai"] else "Human"
                ok = "TAK" if r["correct"] else "**NIE**"
                lines.append(
                    f"| {r['sample_id']} | {actual} | {r['ai_score']:.1f}% "
                    f"| {pred} | {ok} |"
                )
            lines.append("")

        # --- Interpretacja ---
        lines.append("## Interpretacja wynikow")
        lines.append("")

        # Znajdz najlepszy detektor
        best_name = max(
            self.evaluations, key=lambda n: self.evaluations[n].auroc
        )
        best = self.evaluations[best_name]

        lines.append(f"### Najlepszy detektor: {best_name}")
        lines.append("")
        lines.append(
            f"Detektor **{best_name}** osiagnal najwyzsze AUROC = {best.auroc:.3f}."
        )
        lines.append("")

        # Interpretacja Cohen's d
        for name, ev in self.evaluations.items():
            d = abs(ev.cohens_d)
            if d >= 0.8:
                interp = "duzy efekt -- dobra separacja AI vs. human"
            elif d >= 0.5:
                interp = "sredni efekt -- umiarkowana separacja"
            elif d >= 0.2:
                interp = "maly efekt -- slaba separacja"
            else:
                interp = "zaniedbywalny efekt -- brak separacji"
            lines.append(f"- **{name}** Cohen's d = {ev.cohens_d:.2f} ({interp})")

        lines.append("")

        # Ostrzezenia
        lines.append("### Uwagi metodologiczne")
        lines.append("")
        lines.append(
            "1. **TPR @ FPR=1%** jest kluczowa metryka w kontekscie akademickim -- "
            "falszywe oskarzenie studenta o uzycie AI jest bardzo kosztowne."
        )
        lines.append(
            "2. **Cohen's d** mierzy wielkosc efektu, nie istotnosc statystyczna -- "
            "przy malych probach nalezy interpretowac ostroznie."
        )
        lines.append(
            "3. Probki AI sa napisane w **jezyku polskim** w stylu akademickim, "
            "co eliminuje bias jezykowy."
        )
        lines.append(
            "4. Detektory online moga miec zmienne zachowanie w czasie -- "
            "wyniki zalezy traktowac jako punktowa ocene."
        )
        lines.append("")

        # Rekomendacje
        lines.append("## Rekomendacje")
        lines.append("")

        any_good = any(
            ev.auroc >= 0.7 for ev in self.evaluations.values()
        )
        if any_good:
            lines.append(
                "- Detektory z AUROC >= 0.7 moga sluzyc jako wskaznik, "
                "ale **nie jako jedyny dowod** uzycia AI."
            )
        else:
            lines.append(
                "- **Zaden detektor** nie osiagnal AUROC >= 0.7 na polskich tekstach "
                "akademickich -- ich wyniki nie powinny byc traktowane jako wiarygodne."
            )

        for name, ev in self.evaluations.items():
            if ev.confusion.fp > 0:
                fp_rate = ev.confusion.fp / (ev.confusion.fp + ev.confusion.tn)
                lines.append(
                    f"- **{name}** falszywie oskarza {fp_rate:.0%} ludzkich tekstow "
                    f"o bycie AI-generated ({ev.confusion.fp} FP)."
                )

        lines.append("")
        lines.append(
            "Przed podjeciem decyzji na podstawie wynikow detektora nalezy "
            "uwzglednic jego ograniczenia, szczegolnie w kontekscie tekstow "
            "akademickich w jezyku polskim."
        )
        lines.append("")

        return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def select_human_chunks(
    all_chunks: Dict[int, str],
    preferred_ids: Optional[List[int]] = None,
    count: int = 8,
) -> Dict[int, str]:
    """
    Wybierz chunki ludzkie do ewaluacji.
    Preferowane sa chunki ktore wczesniej mialy niski AI% (najbardziej 'ludzkie').
    """
    if preferred_ids:
        selected = {
            cid: all_chunks[cid]
            for cid in preferred_ids
            if cid in all_chunks
        }
        if len(selected) >= count:
            return dict(list(selected.items())[:count])

    # Fallback -- wybierz co count-ty chunk
    sorted_ids = sorted(all_chunks.keys())
    step = max(1, len(sorted_ids) // count)
    chosen_ids = sorted_ids[::step][:count]
    return {cid: all_chunks[cid] for cid in chosen_ids}


def main() -> None:
    """Glowna funkcja ewaluacji."""
    log.info("=" * 60)
    log.info("EWALUACJA DETEKTOROW AI-GENERATED TEXT")
    log.info("Wersja z polskimi probkami AI (fair comparison)")
    log.info("=" * 60)

    # 1. Zaladuj oryginalne chunki
    all_chunks = load_chunks()
    if not all_chunks:
        log.error("Nie znaleziono chunkow! Sprawdz katalog chunks/")
        return

    log.info(f"Zaladowano {len(all_chunks)} chunkow")

    # 2. Wybierz chunki do ewaluacji (preferowane: te ktore mialy niski AI%)
    #    Chunki 1,3,5,8,12,16,20,22 -- rownomiernie rozlozone po tekscie
    preferred = [1, 3, 5, 8, 12, 16, 20, 22]
    human_chunks = select_human_chunks(all_chunks, preferred_ids=preferred, count=8)
    log.info(
        f"Wybrano {len(human_chunks)} chunkow ludzkich: "
        f"{sorted(human_chunks.keys())}"
    )

    # 3. Polskie AI-generated samples (wbudowane)
    ai_samples = POLISH_AI_SAMPLES
    log.info(f"Polskich probek AI: {len(ai_samples)}")

    # 4. Konfiguracja detektorow (darmowe, bez klucza API)
    detector_registry: Dict[str, Callable] = {
        "call_zerogpt": call_zerogpt,
        "call_contentdetector": call_contentdetector,
    }

    log.info(f"Detektory: {list(detector_registry.keys())}")

    # 5. Uruchom ewaluacje
    suite = EvaluationSuite(detectors=list(detector_registry.values()))

    log.info("")
    log.info("Rozpoczynam testy -- to moze potrwac kilka minut...")
    log.info(
        f"Lacznie {len(human_chunks) + len(ai_samples)} probek x "
        f"{len(detector_registry)} detektorow = "
        f"{(len(human_chunks) + len(ai_samples)) * len(detector_registry)} wywolan API"
    )
    log.info("")

    suite.run(
        human_chunks=human_chunks,
        ai_samples=ai_samples,
        delay_between_calls=3.0,
    )

    # 6. Generuj raport
    if not suite.evaluations:
        log.error("Brak wynikow -- wszystkie detektory zawiodly!")
        return

    report_md = suite.generate_report()

    # 7. Przygotuj dane JSON
    json_data = {
        "metadata": {
            "description": "Ewaluacja detektorow AI text -- polskie probki",
            "human_chunks_count": len(human_chunks),
            "human_chunk_ids": sorted(human_chunks.keys()),
            "ai_samples_count": len(ai_samples),
            "ai_sample_ids": list(ai_samples.keys()),
            "detectors": list(detector_registry.keys()),
            "classification_threshold": EvaluationSuite.CLASSIFICATION_THRESHOLD,
        },
        "results": {},
    }

    for name, ev in suite.evaluations.items():
        json_data["results"][name] = {
            "accuracy": round(ev.confusion.accuracy, 4),
            "precision": round(ev.confusion.precision, 4),
            "recall": round(ev.confusion.recall, 4),
            "f1": round(ev.confusion.f1, 4),
            "auroc": ev.auroc,
            "tpr_at_fpr_1pct": ev.tpr_at_fpr_1pct,
            "cohens_d": ev.cohens_d,
            "avg_ai_score_on_human": ev.avg_ai_score_on_human,
            "avg_ai_score_on_ai": ev.avg_ai_score_on_ai,
            "discrimination_gap": ev.discrimination_gap,
            "accuracy_ci_95": [ev.accuracy_ci_lower, ev.accuracy_ci_upper],
            "confusion_matrix": {
                "tp": ev.confusion.tp,
                "fp": ev.confusion.fp,
                "tn": ev.confusion.tn,
                "fn": ev.confusion.fn,
            },
            "roc_points": [
                {"fpr": round(fpr, 4), "tpr": round(tpr, 4)}
                for fpr, tpr in ev.roc_points
            ],
            "per_sample": ev.all_results,
        }

    # 8. Zapisz raporty
    save_json_report(json_data, "evaluation_results.json")
    save_markdown_report(report_md, "RAPORT_EWALUACJI.md")

    # 9. Podsumowanie w konsoli
    log.info("")
    log.info("=" * 60)
    log.info("PODSUMOWANIE")
    log.info("=" * 60)

    for name, ev in suite.evaluations.items():
        log.info(
            f"  {name}: Accuracy={ev.confusion.accuracy:.1%} "
            f"AUROC={ev.auroc:.3f} F1={ev.confusion.f1:.3f} "
            f"Cohen's d={ev.cohens_d:.2f} "
            f"Gap={ev.discrimination_gap:.1f}pp"
        )

    log.info("")
    log.info(f"Raporty zapisane w: {RESULTS_DIR}")
    log.info("  - evaluation_results.json")
    log.info("  - RAPORT_EWALUACJI.md")


if __name__ == "__main__":
    main()
