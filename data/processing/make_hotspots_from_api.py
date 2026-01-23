import json
from datetime import date
import numpy as np
import rasterio
from rasterio.transform import xy
from pystac_client import Client
import planetary_computer as pc

# BBOX Krakowa: (W, S, E, N)
KRAKOW_BBOX = (19.79, 49.96, 20.12, 50.12)

# ilosc hotspotów (punktów) 
TOP_N = 300

# zakres dat do szukania
DATE_RANGE = "2025-06-01/2025-09-15"

# STAC API Planetary Computer
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "landsat-c2-l2"

# Landsat L2 temperatura
TEMP_ASSET_CANDIDATES = ["ST_B10", "ST_TRAD", "ST"]  
def pick_temp_asset(item):
    for name in TEMP_ASSET_CANDIDATES:
        if name in item.assets:
            return name

    raise RuntimeError(f"Nie znalazłem assetu temperatury. Dostępne assety: {list(item.assets.keys())[:30]} ...")

def main():
    catalog = Client.open(STAC_URL)

    search = catalog.search(
        collections=[COLLECTION],
        bbox=KRAKOW_BBOX,
        datetime=DATE_RANGE,
        limit=50,
        query={

            "eo:cloud_cover": {"lt": 20}
        },
    )

    items = list(search.items())
    if not items:
        raise RuntimeError("Nie znalazłem żadnych scen w tym zakresie dat/bbox. Zmień DATE_RANGE albo poluzuj cloud_cover.")

    # zachmurzenie
    def cloud(item):
        return item.properties.get("eo:cloud_cover", 9999)

    items.sort(key=cloud)
    item = items[0]

    scene_date = item.datetime.date().isoformat() if item.datetime else date.today().isoformat()

    # item
    item = pc.sign(item)

    asset_name = pick_temp_asset(item)
    href = item.assets[asset_name].href

   
    with rasterio.open(href) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan

        # top N pikseli
        flat = data.ravel()
        valid_idx = np.where(np.isfinite(flat))[0]
        if valid_idx.size == 0:
            raise RuntimeError("Brak ważnych danych (same NaN/nodata).")

        top_n = min(TOP_N, valid_idx.size)
        valid_vals = flat[valid_idx]
        top_local = np.argpartition(valid_vals, -top_n)[-top_n:]
        top_idx = valid_idx[top_local]

        rows, cols = np.unravel_index(top_idx, data.shape)

        features = []
        for r, c in zip(rows, cols):
            val = float(data[r, c])
            lon, lat = xy(src.transform, r, c, offset="center")
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"temp": val, "date": scene_date}
            })

    geojson = {"type": "FeatureCollection", "features": features}
    out_path = f"data/hotspots_{scene_date}.geojson"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    print("OK ->", out_path)
    print("Scene:", item.id, "cloud_cover:", item.properties.get("eo:cloud_cover"), "asset:", asset_name)

if __name__ == "__main__":
    main()
