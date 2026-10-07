# Predpoveď dažďa – weatherAUS

Klasifikácia `RainTomorrow` (bude zajtra pršať?) pomocou Gaussovho naivného Bayesa a kNN
na datasete [weatherAUS](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package).

## Predspracovanie dát

- **Odstránenie `RISK_MM`**: je to množstvo zrážok na nasledujúci deň, teda by modelu prezradilo odpoveď.
- **`RainTomorrow` ako cieľ `y`**: oddelené od príznakov, Yes = 1, No = 0.
- **Lokalita → zemepisná šírka a dĺžka**: názov mesta nemá číselný význam, súradnice áno.
- **Smer vetra → sin a cos uhla**: samotný uhol by robil zo smerov N (0°) a NNW (337,5°) vzdialené hodnoty, hoci sú susedné.
- **Bezvetrie**: pri rýchlosti vetra 0 smer chýba, lebo neexistuje. Tieto riadky nezahadzujeme, smer nastavíme na sin = cos = 0.
- **Dátum → mesiac ako sin a cos**: ročné obdobie ovplyvňuje dážď a december s januárom zostanú vedľa seba.
- **`RainToday` → 0/1**.
- **`log(1 + x)` pre zrážky a odpar**: väčšina dní má 0 mm, logaritmus zmierni zošikmenie (pomáha Gaussovmu NB).
- **Odstránenie riadkov s chýbajúcimi hodnotami**: klasifikátory s nimi nevedia pracovať.

### Dva varianty dát

Stĺpce `Sunshine`, `Evaporation`, `Cloud9am` a `Cloud3pm` chýbajú v 38–48 % riadkov, ale patria
medzi najsilnejšie prediktory. Preto porovnávame dva varianty:

| Variant | Riadky | Príznaky |
|---|---|---|
| A – bez týchto stĺpcov | 119 571 | 23 |
| B – so všetkými stĺpcami | 58 079 | 27 |

V oboch je dážď nasledujúci deň v 21,9 % prípadov, rovnako ako v pôvodných dátach.

### Normalizácia

Normalizácia na rozsah 0–1 sa robí až počas cross-validácie, iba z trénovacích dát.
Inak by informácia z testovacích dát unikla do trénovania.