from pystac_client import Client
import planetary_computer as pc
from zoneinfo import ZoneInfo

BBOX = (19.80, 49.98, 20.15, 50.12)
START = "2025-06-01"
END   = "2025-10-31"

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=BBOX,
    datetime=f"{START}T00:00:00Z/{END}T23:59:59Z",
    max_items=200,
)

items = list(search.items())
items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 999))

warsaw = ZoneInfo("Europe/Warsaw")

for it in items[:30]:
    cc = it.properties.get("eo:cloud_cover", None)

    if it.datetime:
        dt_utc = it.datetime  
        dt_pl = dt_utc.astimezone(warsaw)
        dt_str = dt_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
        pl_str = dt_pl.strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        dt_str = "?"
        pl_str = "?"

    print(f"{dt_str} | PL: {pl_str} | cloud={cc:>5} | {it.id}")
