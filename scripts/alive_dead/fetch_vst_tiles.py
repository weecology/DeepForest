"""Find and fetch NEON AOP RGB and CHM tiles covering VST dead-tree candidates, for every flight year.

Candidates are individuals ever recorded as standing dead or dead with a broken
bole, with stem coordinates and a recorded height of at least ``--min_height``.
Each stem's 1 km tile is used, plus its neighbour when the stem is within
``--edge_buffer`` m of a tile edge. Tiles are matched by (domain, easting,
northing), so VST sites flown under another AOP site name are covered too.

Tiles already in ``--local_root`` are used in place. Others are downloaded to
``--out_dir/<product>/<year>/<site>/``.

Outputs, in ``--out_dir``:
    tile_manifest.csv  one row per (product, year, tile): local path or URL, size, status
    stem_tiles.csv     one row per (individualID, tile)

Usage:
    NEON_TOKEN=... python fetch_vst_tiles.py --vst_dir <RELEASE dir> --out_dir <dir> [--dry_run]
"""

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://data.neonscience.org/api/v0"
PRODUCTS = {"rgb": "DP3.30010.001", "chm": "DP3.30015.001"}
DEAD = ["Standing dead", "Dead, broken bole"]


def session() -> requests.Session:
    s = requests.Session()
    s.headers["X-API-Token"] = os.environ["NEON_TOKEN"]
    return s


def get(s: requests.Session, url: str, **kw) -> requests.Response:
    """GET with retries that respect NEON rate-limit headers when present."""
    for attempt in range(6):
        try:
            r = s.get(url, timeout=600, **kw)
        except requests.RequestException:
            time.sleep(2**attempt)
            continue
        remaining = r.headers.get("x-ratelimit-remaining")
        if r.status_code == 429 or (remaining is not None and int(remaining) <= 1):
            time.sleep(int(r.headers.get("x-ratelimit-reset", 10)) + 1)
        if r.status_code == 200:
            return r
        if r.status_code in (403, 404):
            r.raise_for_status()
        time.sleep(2**attempt)
    r.raise_for_status()
    return r


def candidate_stems(vst_dir: Path, min_height: float) -> pd.DataFrame:
    """Individuals ever recorded dead (standing or broken bole) with coordinates."""
    m = pd.read_csv(vst_dir / "vst_merged.csv", low_memory=False)
    dead_ids = m.loc[m.plantStatus.isin(DEAD), "individualID"].unique()
    c = m[m.individualID.isin(dead_ids) & m.stemEasting.notna()]
    c = c.groupby("individualID").agg(
        domainID=("domainID", "first"),
        siteID=("siteID", "first"),
        stemEasting=("stemEasting", "last"),
        stemNorthing=("stemNorthing", "last"),
        maxHeight=("height", "max"),
    )
    return c[c.maxHeight >= min_height].reset_index()


def stem_tiles(c: pd.DataFrame, buffer: float) -> pd.DataFrame:
    """Expand each stem to the 1 km tiles within ``buffer`` m of it."""
    offsets = [(dx, dy) for dx in (-buffer, 0, buffer) for dy in (-buffer, 0, buffer)]
    parts = [
        c.assign(
            tileE=((c.stemEasting + dx) // 1000 * 1000).astype(int),
            tileN=((c.stemNorthing + dy) // 1000 * 1000).astype(int),
        )
        for dx, dy in offsets
    ]
    return pd.concat(parts).drop_duplicates(["individualID", "tileE", "tileN"])


def list_remote(s: requests.Session, product: str, domains: set[str]) -> pd.DataFrame:
    """All tiles NEON publishes for a product in the given domains (all releases + provisional)."""
    meta = get(s, f"{API}/products/{PRODUCTS[product]}").json()["data"]["siteCodes"]
    urls = [(site["siteCode"], url) for site in meta for url in site["availableDataUrls"]]

    def one(item):
        site, url = item
        d = get(s, url).json()["data"]
        rows = []
        for f in d["files"]:
            if not f["name"].endswith(".tif"):
                continue
            m = re.search(r"(\d{6})_(\d{7})_(image|CHM)\.tif$", f["name"])
            if m:
                rows.append(
                    {
                        "product": product,
                        "site": site,
                        "month": d["month"],
                        "release": d["release"],
                        "name": f["name"],
                        "url": f["url"],
                        "size": int(f["size"]),
                        "tileE": int(m[1]),
                        "tileN": int(m[2]),
                    }
                )
        return rows

    with ThreadPoolExecutor(4) as pool:
        rows = [r for chunk in pool.map(one, urls) for r in chunk]
    df = pd.DataFrame(rows)
    df["year"] = df.month.str[:4].astype(int)
    dom = {x["siteCode"]: x["domainCode"] for x in get(s, f"{API}/sites").json()["data"]}
    df["domainID"] = df.site.map(dom)
    return df[df.domainID.isin(domains)]


def local_index(local_root: Path, product: str) -> pd.DataFrame:
    """Index of tiles already on disk, keyed by site, year and tile coordinates."""
    rows = []
    for p in local_root.glob(f"*/{PRODUCTS[product]}/neon-aop-*products/*/**/*.tif"):
        m = re.search(r"(\d{6})_(\d{7})_(image|CHM)\.tif$", p.name)
        y = re.search(r"/(\d{4})/", str(p))
        if m and y:
            rows.append(
                {
                    "site": p.parts[len(local_root.parts)],
                    "year": int(y[1]),
                    "tileE": int(m[1]),
                    "tileN": int(m[2]),
                    "local_path": str(p),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["site", "year", "tileE", "tileN"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vst_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--local_root", type=Path, default=Path("/orange/ewhite/NeonData"))
    p.add_argument("--min_height", type=float, default=3.0)
    p.add_argument("--edge_buffer", type=float, default=10.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    s = session()

    stems = stem_tiles(candidate_stems(a.vst_dir, a.min_height), a.edge_buffer)
    stems.to_csv(a.out_dir / "stem_tiles.csv", index=False)
    need = stems[["domainID", "tileE", "tileN"]].drop_duplicates()
    print(
        f"{stems.individualID.nunique():,} stems -> {len(need):,} unique 1 km tiles",
        flush=True,
    )

    manifests = []
    for product in PRODUCTS:
        remote = list_remote(s, product, set(need.domainID)).merge(
            need, on=["domainID", "tileE", "tileN"]
        )
        # One tile per (product, year, tile): prefer released over provisional, then latest month.
        remote = remote.sort_values(["release", "month"]).drop_duplicates(
            ["domainID", "year", "tileE", "tileN"], keep="last"
        )
        local = local_index(a.local_root, product)
        remote = remote.merge(local, on=["site", "year", "tileE", "tileN"], how="left")
        dest = a.out_dir / product / remote.year.astype(str) / remote.site / remote.name
        remote["dest"] = np.where(
            remote.local_path.notna(), remote.local_path, dest.astype(str)
        )
        remote["status"] = np.where(remote.local_path.notna(), "local", "pending")
        exists = remote.dest.map(lambda d: Path(d).exists())
        remote.loc[exists & (remote.status == "pending"), "status"] = "downloaded"
        manifests.append(remote)
        todo = remote[remote.status == "pending"]
        print(
            f"{product}: {len(remote):,} tile-years ({remote.year.min()}-{remote.year.max()}); "
            f"local {(remote.status == 'local').sum():,}; to download {len(todo):,} "
            f"({todo['size'].sum() / 1e9:,.1f} GB)",
            flush=True,
        )
    man = pd.concat(manifests, ignore_index=True)
    man.to_csv(a.out_dir / "tile_manifest.csv", index=False)
    if a.dry_run:
        return

    def fetch(row) -> str:
        out = Path(row.dest)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        url = row.url
        for attempt in range(4):
            try:
                with s.get(url, stream=True, timeout=600) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(1 << 22):
                            fh.write(chunk)
                if tmp.stat().st_size == row["size"]:
                    tmp.rename(out)
                    return "downloaded"
            except requests.RequestException:
                pass
            time.sleep(2**attempt)
            # Signed storage URLs expire; refresh from the API listing.
            url = next(
                (
                    f["url"]
                    for f in get(
                        s, f"{API}/data/{PRODUCTS[row['product']]}/{row.site}/{row.month}"
                    ).json()["data"]["files"]
                    if f["name"] == row["name"]
                ),
                url,
            )
        return "failed"

    todo = man[man.status == "pending"]
    with ThreadPoolExecutor(a.workers) as pool:
        for i, (idx, st) in enumerate(
            zip(todo.index, pool.map(fetch, (r for _, r in todo.iterrows())), strict=True)
        ):
            man.loc[idx, "status"] = st
            if i % 50 == 0:
                print(f"{i + 1}/{len(todo)} {st}", flush=True)
                man.to_csv(a.out_dir / "tile_manifest.csv", index=False)
    man.to_csv(a.out_dir / "tile_manifest.csv", index=False)
    print(man.groupby(["product", "status"]).size().to_string())


if __name__ == "__main__":
    main()
