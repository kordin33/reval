# INSTRUKCJA TESTOWANIA PRACY NA DETEKTORY AI

## Jak zaczac?

### Import nowej pracy:
```bash
cd E:\ai-text-detection-academic
python main.py --input sciezka/do/pracy.tex         # LaTeX
python main.py --input sciezka/do/pracy.txt         # Zwykly tekst
python main.py --input praca.tex --full             # Import + pelna analiza
```

Tekst zostanie wyekstrahowany i podzielony na chunki gotowe do testowania.

**Lokalizacja projektu:**
```
E:\ai-text-detection-academic\
├── full_text.txt        # Pelny tekst (~58k znakow)
├── chunks\              # Chunki po ~2500 znakow (po imporcie pracy)
│   ├── chunk_01.txt ... chunk_24.txt
├── main.py              # Centralny punkt wejscia (CLI)
├── config.py            # Konfiguracja (thresholds, API keys, endpointy)
├── utils.py             # Wspolne narzedzia
├── binoculars_detector.py  # Binoculars (lokalna, zero-shot, najlepsza dla polskiego!)
├── evaluation.py        # Ewaluacja detektorow (AUROC, F1, confusion matrix)
├── ensemble_scorer.py   # Wagowana agregacja wielu detektorow
├── style_analyzer.py    # Analiza stylu + sugestie poprawek
├── local_detector.py    # Modele HuggingFace (RoBERTa)
├── modern_detector.py   # Detektory API (Originality, GPTZero, Sapling)
├── jsa_analysis.py      # Symulacja JSA (perplexity)
└── results\             # Wyniki testow
```

---

## OPCJA 1: Pelne skanowanie (zalecane)

```bash
cd E:\ai-text-detection-academic
python main.py --full
```

To uruchomi: detekcje na wszystkich detektorach, ewaluacje, analize stylu.

### Inne komendy:

```bash
python main.py --detect              # Tylko detekcja AI
python main.py --evaluate            # Ewaluacja dokladnosci detektorow
python main.py --analyze-style       # Analiza stylu + sugestie
python main.py --binoculars          # Tylko Binoculars (lokalna, GPU)
```

---

## OPCJA 2: Binoculars (najlepsza dla polskiego tekstu!)

Binoculars to metoda zero-shot - **nie wymaga treningu na polskich danych**. Uzywa dwoch modeli LLM do porownania cross-perplexity. Dziala natywnie na kazdym jezyku.

```bash
python binoculars_detector.py
```

**Wymagania:** GPU z min. 4GB VRAM, PyTorch + transformers

---

## OPCJA 3: Testy reczne (bez instalacji)

Otwierz strone i wklej fragmenty tekstu:

### 1. Copyleaks (darmowe, bez logowania)
https://copyleaks.com/ai-content-detector
- Do 25,000 znakow, wspiera 30+ jezykow (w tym polski!)

### 2. ZeroGPT (darmowe, bez logowania)
https://zerogpt.com
- Do 15,000 znakow

### 3. Sapling AI Detector (darmowe)
https://sapling.ai/ai-content-detector

### 4. GPTZero (darmowe, 10k slow/miesiac)
https://gptzero.me
- Wymaga rejestracji, najdokladniejszy na angielskim

### 5. ContentDetector.AI (darmowe)
https://contentdetector.ai/

### 6. Scribbr (darmowe)
https://www.scribbr.com/ai-detector/

---

## OPCJA 4: Testy przez API

### Klucze API (opcjonalne, dla wiekszej dokladnosci):

```bash
# W PowerShell:
$env:SAPLING_API_KEY = "twoj_klucz"      # https://sapling.ai/user/settings
$env:GPTZERO_API_KEY = "twoj_klucz"      # https://gptzero.me
$env:ORIGINALITY_API_KEY = "twoj_klucz"  # https://originality.ai (platne)
$env:HF_TOKEN = "twoj_token"             # https://huggingface.co/settings/tokens

# W bash:
export SAPLING_API_KEY="twoj_klucz"
```

---

## REKOMENDOWANY PLAN TESTOW

### Przetestuj minimum 5 fragmentow:

Wybierz chunki rownomiernie rozlozone po pracy, np:
- Poczatek (wstep/motywacja)
- 1/4 pracy (teoria/przeglad literatury)
- Polowa (metodologia/implementacja)
- 3/4 (wyniki)
- Koniec (wnioski)

### Na minimum 3 detektorach:
- Binoculars (lokalna) - najlepsza dla polskiego
- ZeroGPT (darmowa, online)
- Copyleaks (darmowy, online, wspiera polski)

---

## JAK INTERPRETOWAC WYNIKI

| Zakres    | Werdykt              | Co robic                              |
|-----------|----------------------|---------------------------------------|
| < 20%     | Bardzo bezpieczne    | Nic nie zmieniaj                      |
| 20-40%    | Raczej bezpieczne    | Drobne poprawki opcjonalne            |
| 40-60%    | Wymaga uwagi         | Popraw styl flagowanych fragmentow    |
| 60-80%    | Wysokie ryzyko       | Przepisz flagowane zdania             |
| > 80%     | Krytyczne ryzyko     | Przepisz caly fragment od nowa        |

### Wazne zastrzezenia:
- Detektory AI sa **zawodne** - zaden nie ma 100% dokladnosci
- Tekst akademicki w jezyku polskim czesto daje **falszywe alarmy**
- Rozne detektory daja **rozne wyniki** - dlatego uzywaj ensemble
- **Zaden detektor nie jest dowodem** - to narzedzie przesiewowe
- JSA (polski system) **nie ma jeszcze dedykowanego modulu AI** (stan 2025-2026)

---

## JESLI WYNIKI SA WYSOKIE (>50%)

Uzyj analizatora stylu:

```bash
python style_analyzer.py chunks/chunk_01.txt
```

Dostaniesz konkretne sugestie per-zdanie, np.:
- "To zdanie ma 38 slow - podziel na 2 krotsze"
- "Zamien 'nalezy podkreslic' na 'warto zauwazyc'"
- "Uzyj strony czynnej zamiast biernej"

Wiecej szczgolow: patrz `JAK_OBNIZYC_WYNIK_JSA.md`

---

## NASTEPNE KROKI

1. Uruchom `python main.py --full`
2. Przejrzyj raport w `results/`
3. Jesli srednia >50%, uzyj `style_analyzer.py` na flagowanych chunkach
4. Popraw tekst wedlug sugestii
5. Przetestuj ponownie
