# Urban Heat Island – Kraków (Landsat ST hotspots)

Statyczna aplikacja web pokazująca punkty hotspotów temperatury powierzchni (Surface Temperature, ST) dla Krakowa na mapie Leaflet.
Dane pochodzą z Landsat Collection 2 Level-2 (ST_B10) przez Microsoft Planetary Computer (STAC).

## Co jest w repo
- `index.html` – frontend (Leaflet + OSM + warstwa satelitarna + legenda).
- `data/hotspots_YYYY-MM-DD.geojson` – wygenerowane punkty hotspotów.
- `data/processing/make_hotspots_from_api.py` – generator GeoJSON z Planetary Computer.
- `data/processing/find_scenes.py` – pomocniczy plik do wyszukiwania scen na podstawie daty i zachmurzenia.
- `.github/workflows/deploy.yml` – CI/CD deploy do Azure Storage Static Website ($web) przez azcopy (SAS).
- `.github/workflows/generate-hotspots.yml` – ręczny job generowania hotspotów i upload do Azure.
- `iac/main.bicep` – IaC do utworzenia Storage Account i włączenia static website.
- `docs/PROJECT_DOC.md` – dokumentacja projektu (architektura, przepływy, koszty, monitoring, cleanup).

---

## Uruchomienie lokalne
Visual Studio Code -> index.html (Live Server)
