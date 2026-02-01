import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from pystac_client import Client
import planetary_computer as pc


# --- PROJ fix ----------------
venv = os.environ.get("VIRTUAL_ENV")
if venv:
    proj_data = Path(venv) / "Lib" / "site-packages" / "rasterio" / "proj_data"
    if proj_data.exists():
        os.environ["PROJ_LIB"] = str(proj_data)
        os.environ.pop("PROJ_DATA", None)


# --------- konfiguracja ---------
DATE = os.getenv("DATE", "2025-07-24")
BBOX = tuple(map(float, os.getenv("BBOX", "19.80,49.98,20.15,50.12").split(",")))  # (minLon, minLat, maxLon, maxLat)

OUT_DIR = os.getenv("OUT_DIR", "data")
OUT_PATH = os.path.join(OUT_DIR, f"hotspots_{DATE}.geojson")

N_POINTS = int(os.getenv("N_POINTS", "10000"))  # ile losowych pikseli próbkujemy do selekcji hotspotów
TOP_K = int(os.getenv("TOP_K", "5000"))         # ile najcieplejszych punktów zapisujemy w GeoJSON
MAX_CLOUD = float(os.getenv("MAX_CLOUD", "20")) # filtr po metadanych sceny (eo:cloud_cover)

DEBUG = os.getenv("DEBUG", "0").strip() in ("1", "true", "True", "YES", "yes")


def dprint(*args):
    if DEBUG:
        print(*args)


def get_band_metadata(asset) -> Dict[str, Any]:
    """
    Zwraca pierwszy wpis z raster:bands (jeśli jest), inaczej {}.
    """
    bands = None
    if hasattr(asset, "extra_fields"):
        bands = asset.extra_fields.get("raster:bands")
    if isinstance(bands, list) and bands and isinstance(bands[0], dict):
        return bands[0]
    return {}


def get_scale_offset_from_asset(asset) -> Tuple[Optional[float], Optional[float]]:
    """
    Planetary Computer / STAC często trzyma scale/offset w asset.extra_fields['raster:bands'].
    Zwraca (scale, offset) lub (None, None).
    """
    band = get_band_metadata(asset)
    scale = band.get("scale")
    offset = band.get("offset")
    try:
        scale_f = float(scale) if scale is not None else None
        offset_f = float(offset) if offset is not None else None
    except Exception:
        return None, None
    return scale_f, offset_f


def looks_like_surface_temperature(asset_name: str, asset) -> bool:
    """
    Heurystyka bezpieczeństwa:
    - odrzucamy radiancję i pochodne (trad/urad/drad) po nazwie
    - akceptujemy, jeśli metadane sugerują temperaturę w K (unit=kelvin/K)
      albo jeśli scale/offset są bliskie standardowym dla Landsat C2 L2 ST.
    """
    lname = asset_name.lower()

    if any(x in lname for x in ["trad", "urad", "drad", "radiance"]):
        return False

    band = get_band_metadata(asset)
    unit = str(band.get("unit") or "").lower()
    scale, offset = get_scale_offset_from_asset(asset)

    if unit in ["kelvin", "k"]:
        return True

    if scale is not None and offset is not None:
        if abs(scale - 0.00341802) < 0.0005 and abs(offset - 149.0) < 2.0:
            return True

    return False


def pick_temp_asset(item) -> Optional[str]:
    """
    W Planetary Computer Landsat ST często jest pod nazwą "lwir11" (plik *_ST_B10.TIF).
    Wybieramy tylko asset, który wygląda jak Surface Temperature.

    Zasada:
    1) Preferuj "lwir11", jeśli przechodzi walidację ST.
    2) W przeciwnym razie przeszukaj wszystkie assety i wybierz pierwszy, który przechodzi walidację.
    """
    if "lwir11" in item.assets and looks_like_surface_temperature("lwir11", item.assets["lwir11"]):
        return "lwir11"

    for name, asset in item.assets.items():
        if looks_like_surface_temperature(name, asset):
            return name

    return None


def dn_to_celsius(dn: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """
    Zakładamy, że po skalowaniu dostajemy Kelviny (typowe dla Landsat ST).
    """
    kelvin = dn.astype(np.float32) * scale + offset
    return kelvin - 273.15


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=BBOX,
        datetime=f"{DATE}T00:00:00Z/{DATE}T23:59:59Z",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        max_items=25,
    )

    items = list(search.items())
    if not items:
        raise RuntimeError(
            "Nie znalazłem sceny Landsat dla tej daty/BBOX. "
            "Spróbuj inną datę, większy BBOX albo zwiększ MAX_CLOUD."
        )

    # scena o najmniejszym zachmurzeniu (eo:cloud_cover)
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 999.0))
    item = items[0]

    dprint("Scene chosen:", item.id, "cloud_cover:", item.properties.get("eo:cloud_cover"))
    dprint("Assets:", list(item.assets.keys()))

    asset_name = pick_temp_asset(item)
    if asset_name is None:
        if DEBUG:
            print("\n=== DEBUG: raster:bands (wybrane) ===")
            for k in ["lwir11", "trad", "urad", "drad", "emis", "emsd"]:
                if k in item.assets:
                    a = item.assets[k]
                    bands = getattr(a, "extra_fields", {}).get("raster:bands")
                    print(f"\nASSET: {k}")
                    print("href:", a.href)
                    print("raster:bands:", bands[:1] if isinstance(bands, list) else bands)

        raise RuntimeError(
            "Nie znalazłem wiarygodnego assetu Surface Temperature (ST). "
            "To celowe (żeby nie udawać °C z radiancji). "
            f"Dostępne assety: {list(item.assets.keys())}"
        )

    asset = item.assets[asset_name]
    band_meta = get_band_metadata(asset)
    unit_from_stac = str(band_meta.get("unit") or "")
    scale, offset = get_scale_offset_from_asset(asset)

    if not looks_like_surface_temperature(asset_name, asset):
        raise RuntimeError(
            f"Wybrany asset nie wygląda jak Surface Temperature: {asset_name}. "
            f"stac_unit={unit_from_stac} meta={band_meta}"
        )

    if scale is None or offset is None:
        raise RuntimeError(
            f"Brak scale/offset dla assetu ST ({asset_name}). "
            f"stac_unit={unit_from_stac} meta={band_meta}. "
            "Nie generuję pliku, bo nie da się pewnie policzyć temperatury."
        )

    href = pc.sign(asset.href)

    with rasterio.open(href) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata

        mask = np.isfinite(arr)

        if nodata is not None:
            mask &= (arr != nodata)

        mask &= (arr > 0)

        valid_count = int(mask.sum())
        if valid_count < 50:
            raise RuntimeError(f"Za mało poprawnych pikseli po maskowaniu: {valid_count}")

        # Losowe próbkowanie pikseli
        rng = np.random.default_rng(0)
        flat_idx = np.flatnonzero(mask)
        if len(flat_idx) > N_POINTS:
            flat_idx = rng.choice(flat_idx, size=N_POINTS, replace=False)

        rows, cols = np.unravel_index(flat_idx, arr.shape)
        sampled_dn = arr[rows, cols]

        # Konwersja DN -> °C
        temps_c = dn_to_celsius(sampled_dn, scale, offset)
        unit = "C"

        # TOP_K najcieplejszych
        if TOP_K > 0 and TOP_K < len(temps_c):
            top_idx = np.argpartition(temps_c, -TOP_K)[-TOP_K:]
            rows = rows[top_idx]
            cols = cols[top_idx]
            temps_c = temps_c[top_idx]

        # piksel -> współrzędne 
        xs, ys = rasterio.transform.xy(ds.transform, rows, cols, offset="center")

        if ds.crs is None:
            raise RuntimeError("Raster nie ma CRS (ds.crs=None).")

        lons, lats = warp_transform(ds.crs, "EPSG:4326", xs, ys)

    # statystyki
    tmin = float(np.nanmin(temps_c))
    tmax = float(np.nanmax(temps_c))
    tavg = float(np.nanmean(temps_c))

    print(f"asset={asset_name} unit={unit} stac_unit={unit_from_stac} scale={scale} offset={offset}")
    print(f"stats: min={tmin:.2f} avg={tavg:.2f} max={tmax:.2f} n={len(temps_c)}")
    print("Scene:", item.id, "cloud_cover:", item.properties.get("eo:cloud_cover"))

    features: List[Dict[str, Any]] = []
    for lon, lat, t in zip(lons, lats, temps_c):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "temp": float(np.round(t, 1)),
                    "unit": unit,
                    "date": DATE,
                    "scene": item.id,
                    "asset": asset_name,
                },
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            }
        )

    fc: Dict[str, Any] = {
        "type": "FeatureCollection",
        "properties": {
            "date": DATE,
            "scene": item.id,
            "asset": asset_name,
            "unit": unit,
            "stac_unit": unit_from_stac,
            "scale": scale,
            "offset": offset,
            "min": float(np.round(tmin, 2)),
            "avg": float(np.round(tavg, 2)),
            "max": float(np.round(tmax, 2)),
            "n": int(len(features)),
        },
        "features": features,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)

    print(f"OK -> {OUT_PATH}")


if __name__ == "__main__":
    main()
