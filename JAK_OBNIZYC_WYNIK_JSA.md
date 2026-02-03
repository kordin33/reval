# JAK OBNIZYC WYNIK W JSA / POLSKICH DETEKTORACH AI

## Stan wiedzy (2025-2026)

### JSA (Jednolity System Antyplagiatowy)
- JSA to obowiazkowy system antyplagiatowy na polskich uczelniach (od 2019, MEiN/OPI PIB)
- Sprawdza prace w Ogolnopolskim Repozytorium Prac Dyplomowych (ORPPD)
- **JSA MA juz modul detekcji AI** (od wersji 2.15.0, grudzien 2023) - bazuje na **perplexity**
- Metoda: "im wieksza regularnosc tekstu, tym wieksze prawdopodobienstwo ze tekst jest z modelu jezykowego"
- **UWAGA**: JSA sam przyznaje ze ma bledy (false positives i false negatives)
- Definicje, regulaminy, schematy moga byc falgowane mimo ze sa pisane recznie (naturalna niska perplexity)

### Co mierza detektory AI?
- **Perplexity** - jak bardzo tekst "zaskakuje" model jezykowy. Niska perplexity = przewidywalny = AI-like
- **Burstiness** - zmiennosc zlozonosci zdan. Ludzie pisza "burstowo" (krotkie i dlugie zdania na przemian). AI jest rowne
- **Cross-perplexity** (Binoculars) - czy rozne modele AI "zgadzaja sie" co do tekstu. Jesli tak = prawdopodobnie AI
- **Stylometria** - porownanie stylu z bazowym profilem autora

### Dlaczego tekst akademicki jest problematyczny?
Tekst akademicki naturalnie ma niska perplexity (formalny styl, powtarzalne konstrukcje). Dlatego nawet teksty pisane recznie przed era AI moga byc flagowane!

---

## SPRAWDZONE METODY OBNIZENIA WYNIKU

### 1. ZWIEKSZ ZMIENNOSC ZDAN

**Unikaj:**
```
Zastosowanie uczenia maszynowego w analizie danych stalo sie 
powszechnym podejsciem w badaniach naukowych. Metody te wykorzystuja
zaawansowane architektury neuronowe. Sieci neuronowe...
```

**Pisz:**
```
Uczenie maszynowe zmienilo podejscie do analizy danych - i to 
blyskawicznie. Co to oznacza w praktyce? Nawet skomplikowane problemy 
mozna rozwiazac automatycznie. (Ale nie zawsze tak prosto, jak sie wydaje.)
```

### 2. MIESZAJ DLUGOSCI ZDAN

**Unikaj:** Same zlozone zdania 25+ slow
**Pisz:** Mieszaj: krotkie (5 slow). Srednie (12 slow). Czasem dluzsze konstrukcje z kilkoma zdaniami podrzednymi, co daje naturalny rytm.

**Cel:** Wariancja dlugosci zdan > 20 (sprawdz w `style_analyzer.py`)

### 3. DODAJ OSOBISTE ELEMENTY

```
Podczas implementacji natknąłem się na...
W trakcie testow okazalo sie, ze...
Co zaskakujace, model...
Warto zauwazyc, ze moje wyniki roznia sie od...
Szczerze mowiac, spodziewalem sie lepszych wynikow.
```

### 4. UNIKAJ SZABLONOWYCH FRAZ

| Flagowane (niskie perplexity) | Lepsze (wyzsze perplexity) |
|-------------------------------|----------------------------|
| Z punktu widzenia... | Patrzac na to z perspektywy... |
| Nalezy podkreslic, ze... | Warto zauwazyc, ze... / Ciekawe jest to, ze... |
| W niniejszej pracy... | W tej pracy... / Tutaj pokazuje... |
| Podsumowujac, mozna stwierdzic... | Krotko mowiac... |
| W zwiazku z powyzszym... | Dlatego... / W efekcie... |
| Majac na uwadze... | Biorac pod uwage... / Skoro tak... |
| Istotnym aspektem jest... | Wazne jest to, ze... |
| Wykazano, ze... | Okazuje sie, ze... |
| Przeprowadzona analiza... | Analiza pokazala... |
| Na podstawie przeprowadzonych badan... | Badania pokazaly... / Z testow wynika... |
| Zagadnienie to... | Ten problem... / Ta kwestia... |

### 5. DODAJ PYTANIA RETORYCZNE

```
Dlaczego to takie wazne? Bo...
Czy to oznacza, ze detektory sa bezuzyteczne? Nie do konca.
Jak to wplywa na praktyczne zastosowania?
Co z tego wynika dla dalszych badan?
```

### 6. UZYJ IDIOMOW I KOLOKWIALIZMOW (gdzie pasuje)

```
...ale to dopiero wierzcholek gory lodowej.
...co okazalo sie strzalem w dziesiatke.
...model spisal sie zaskakujaco dobrze.
...to zmienia postac rzeczy.
```

### 7. ZAMIEN STRONE BIERNA NA CZYNNA

| Strona bierna (AI-like) | Strona czynna (ludzka) |
|-------------------------|------------------------|
| Zostaly przeprowadzone badania... | Przeprowadzilem badania... |
| Model zostal wytrenowany... | Wytrenowalem model... |
| Wyniki zostaly przedstawione... | Przedstawiam wyniki... |
| Analiza zostala wykonana... | Przeanalizowałem... |

### 8. DODAJ KONKRETNE SZCZEGOLY

Detektory AI rzadko flaguja tekst z konkretnymi szczegolami:
- Daty eksperymentow: "Trening trwal 4h 23min na RTX 4060 Ti (12 kwietnia 2026)"
- Nazwy plikow: "w pliku train_model.py, linia 127"
- Specyficzne wartosci: "accuracy spadla z 94.3% do 87.1% po 15 epoce"
- Osobiste obserwacje: "zauwazam, ze model ma problem z twarzami pod katem >45 stopni"

---

## NARZEDZIA DO SPRAWDZENIA

### Lokalne (najlepsze dla polskiego!):
1. **Pangram EditLens** - `python pangram_detector.py` (ICLR 2026, open-source, F1=1.0!)
2. **Binoculars** - `python binoculars_detector.py` (zero-shot, language-agnostic, GPU)
3. **Style Analyzer** - `python style_analyzer.py` (analiza stylu + konkretne sugestie)

### Online (darmowe):
1. **Grammarly** - https://www.grammarly.com/ai-detector (#1 na RAID benchmark 2026!)
2. **Copyleaks** - https://copyleaks.com/ai-content-detector (wspiera polski!)
3. **GPTZero** - https://gptzero.me (99% accuracy, 10k slow/m free)
4. **ZeroGPT** - https://zerogpt.com (darmowy, bez logowania)
5. **Pangram** - https://www.pangram.com (wspiera 20+ jezykow)
6. **Scribbr** - https://www.scribbr.com/ai-detector/

### Platne (najdokladniejsze):
1. **Grammarly API** - #1 RAID, API: api.grammarly.com
2. **Originality.ai** - ~$0.01/100 slow, 96% accuracy
3. **Winston AI** - gowinston.ai, 99.98% claimed
4. **Plag.pl** - ~10 zl, podobny do JSA - https://sprawdz-prace.plagiat.pl

---

## O WATERMARKINGU

Niektore modele AI wbudowuja "znaki wodne" w generowany tekst:
- **Google SynthID Text** - wbudowany w Gemini. Statistyczny bias w doborze tokenow.
- **OpenAI** - ma technologie watermarkingu, ale nie wdrozyl jej w pelni
- **Meta (Llama)** - modele open-source, brak watermarkingu

Watermarking mozna pokonac przez parafrazowanie lub tlumaczenie. Ale warto o nim wiedziec.

---

## STYLOMETRIA - POROWNANIE Z WLASNYM STYLEM

Niektore uczelnie porownuja styl pracy z poprzednimi tekstami studenta. Jesli zwykle piszesz krotkie zdania, a praca ma same dlugie - to podejrzane.

**Jak sie zabezpieczyc:**
- Zachowaj historie edycji (Git, wersje plikow)
- Zachowaj notatki, szkice, brudnopisy
- Pisz w swoim naturalnym stylu
- Jesli uzywales AI do pomocy (tlumaczenia, korekta) - udokumentuj to

---

## NAJWAZNIEJSZE

1. **Porozmawiaj z promotorem** PRZED oddaniem - wiele uczelni ma wlasna polityke dot. AI
2. **Zaden detektor AI nie jest dowodem** - to narzedzie przesiewowe, nie wyrok
3. **Zachowaj dowody procesu pisania** - Git, wersje, notatki
4. **Tekst akademicki naturalnie "wyglada jak AI"** - niech Cie to nie przerazada
5. **JSA glownie sprawdza plagiaty** - detekcja AI to dodatkowa funkcja w fazie rozwoju

*Pamietaj: Nikt nie powinien byc karany za pisanie w formalnym stylu akademickim.
Jesli Twoja praca zostanie niesluszne flagowana, masz prawo do wyjasnien.*
