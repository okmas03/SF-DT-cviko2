from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "weatherAUS.csv"

##Lokacia na zemepisnej sirke a dlzke z nazvu mesta
LOCATION_COORDS = {
    "Albury": (-36.08, 146.92), "BadgerysCreek": (-33.88, 150.75),
    "Cobar": (-31.50, 145.84), "CoffsHarbour": (-30.30, 153.11),
    "Moree": (-29.47, 149.84), "Newcastle": (-32.93, 151.78),
    "NorahHead": (-33.28, 151.57), "NorfolkIsland": (-29.04, 167.95),
    "Penrith": (-33.75, 150.69), "Richmond": (-33.60, 150.75),
    "Sydney": (-33.87, 151.21), "SydneyAirport": (-33.94, 151.18),
    "WaggaWagga": (-35.12, 147.37), "Williamtown": (-32.80, 151.84),
    "Wollongong": (-34.42, 150.89), "Canberra": (-35.28, 149.13),
    "Tuggeranong": (-35.42, 149.09), "MountGinini": (-35.53, 148.77),
    "Ballarat": (-37.56, 143.85), "Bendigo": (-36.76, 144.28),
    "Sale": (-38.10, 147.07), "MelbourneAirport": (-37.67, 144.84),
    "Melbourne": (-37.81, 144.96), "Mildura": (-34.19, 142.16),
    "Nhil": (-36.33, 141.65), "Portland": (-38.34, 141.60),
    "Watsonia": (-37.71, 145.08), "Dartmoor": (-37.92, 141.27),
    "Brisbane": (-27.47, 153.03), "Cairns": (-16.92, 145.77),
    "GoldCoast": (-28.02, 153.40), "Townsville": (-19.26, 146.82),
    "Adelaide": (-34.93, 138.60), "MountGambier": (-37.83, 140.78),
    "Nuriootpa": (-34.47, 139.00), "Woomera": (-31.20, 136.83),
    "Albany": (-35.02, 117.88), "Witchcliffe": (-34.03, 115.10),
    "PearceRAAF": (-31.67, 116.01), "PerthAirport": (-31.94, 115.97),
    "Perth": (-31.95, 115.86), "SalmonGums": (-32.98, 121.64),
    "Walpole": (-34.98, 116.73), "Hobart": (-42.88, 147.33),
    "Launceston": (-41.43, 147.14), "AliceSprings": (-23.70, 133.88),
    "Darwin": (-12.46, 130.84), "Katherine": (-14.47, 132.26),
    "Uluru": (-25.34, 131.04),
}
 
# 16 smerov vetra, každý o 22,5° ďalej (N = 0°, E = 90°, ...)
WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
WIND_ANGLE = {d: i * 22.5 for i, d in enumerate(WIND_DIRS)}
WIND_COLS = ["WindGustDir", "WindDir9am", "WindDir3pm"]
 
# Stĺpce, ktorým chýba ~38–48 % hodnôt
SPARSE_COLS = ["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"]
 
# Vybrané príznaky pre kNN: kNN váži každý stĺpec rovnako, takže odstraňujeme
# duplicitné (4 teploty -> 1, 2 tlaky -> 1, 3 smery vetra -> nárazový) a
# ponechávame tie, ktoré najviac súvisia s dažďom.
KNN_FEATURES = [
    "Humidity3pm", "Pressure3pm", "Temp3pm", "Rainfall", "RainToday",
    "WindGustSpeed", "WindGustDir_sin", "WindGustDir_cos",
    "Lat", "Lon", "Month_sin", "Month_cos",
]
KNN_FEATURES_SPARSE = ["Sunshine", "Cloud3pm"]  # pridajú sa, ak sú v dátach
 
 
def knn_features(X: pd.DataFrame) -> list:
    return KNN_FEATURES + [c for c in KNN_FEATURES_SPARSE if c in X.columns]
 
 
def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["NA"])
 
 
def preprocess(df: pd.DataFrame, drop_sparse: bool = True):
    """
    drop_sparse=True  -> zahodí SPARSE_COLS, potom riadky s NA (~120k riadkov, 44 lokalít)
    drop_sparse=False -> ponechá ich, zahodí riadky s NA (~58k riadkov, 26 lokalít)
    """
    df = df.copy()
 
    # 1) Cieľ a únik informácie: RISK_MM = zrážky zajtra -> preč, RainTomorrow -> y
    df = df.drop(columns=["RISK_MM"])
 
    if drop_sparse:
        df = df.drop(columns=SPARSE_COLS)
 
    # 2) Bezvetrie: pri rýchlosti 0 dataset smer neuvádza (NA). Nie je to chýbajúci
    #    údaj, smer jednoducho neexistuje -> označíme "CALM" (sin = cos = 0).
    for speed, direction in (("WindSpeed9am", "WindDir9am"), ("WindSpeed3pm", "WindDir3pm")):
        calm = (df[speed] == 0) & df[direction].isna()
        df.loc[calm, direction] = "CALM"
 
    # 3) Neúplné riadky
    df = df.dropna().reset_index(drop=True)
 
    y = (df.pop("RainTomorrow") == "Yes").astype(int)
 
    # 4) Lokalita -> zemepisná šírka a dĺžka
    coords = df.pop("Location").map(LOCATION_COORDS)
    df["Lat"] = coords.str[0]
    df["Lon"] = coords.str[1]
 
    # 5) Smer vetra -> uhol -> sin/cos (0° a 337,5° sú si blízko)
    for col in WIND_COLS:
        direction = df.pop(col)
        rad = np.deg2rad(direction.map(WIND_ANGLE))
        calm = direction == "CALM"
        df[f"{col}_sin"] = np.where(calm, 0.0, np.sin(rad))
        df[f"{col}_cos"] = np.where(calm, 0.0, np.cos(rad))
 
    # 6) Dátum -> mesiac ako cyklická premenná (ročné obdobie)
    month = pd.to_datetime(df.pop("Date")).dt.month
    df["Month_sin"] = np.sin(2 * np.pi * month / 12)
    df["Month_cos"] = np.cos(2 * np.pi * month / 12)
 
    # 7) Binárne kódovanie
    df["RainToday"] = (df["RainToday"] == "Yes").astype(int)
 
    # 8) Zošikmené zrážky/odpar -> log(1 + x), pomáha hlavne Gaussovmu NB
    df["Rainfall"] = np.log1p(df["Rainfall"])
    if "Evaporation" in df:
        df["Evaporation"] = np.log1p(df["Evaporation"])
 
    return df.astype(float), y
 
 
if __name__ == "__main__":
    raw = load_raw()
    print(f"Pôvodný dataset: {raw.shape[0]} riadkov, {raw.shape[1]} stĺpcov\n")
    for drop_sparse in (True, False):
        X, y = preprocess(raw, drop_sparse=drop_sparse)
        label = "bez riedkych stĺpcov" if drop_sparse else "so všetkými stĺpcami"
        print(f"Variant {label}: X = {X.shape}, podiel 'Yes' = {y.mean():.3f}")
        print(f"  príznaky: {list(X.columns)}\n")
 