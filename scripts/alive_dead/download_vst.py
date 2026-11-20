"""Download NEON Vegetation Structure (DP1.10098.001) tables for one release via the NEON API.

Downloads only the basic-package CSVs that are needed and stacks them across
sites and months. ``vst_mappingandtagging`` is published as a cumulative
site-level file in every month, so only the newest copy per site is kept.

Requires a NEON API token in ``NEON_TOKEN``. Unauthenticated bulk requests are
rate limited and then blocked.

Usage:
    NEON_TOKEN=... python download_vst.py --out_dir <dir> [--release RELEASE-2026]
"""

import argparse
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

API = "https://data.neonscience.org/api/v0"
DPID = "DP1.10098.001"
TABLES = ["vst_mappingandtagging", "vst_apparentindividual", "vst_perplotperyear"]
SITE_LEVEL = {"vst_mappingandtagging"}


def get(session: requests.Session, url: str, **kw) -> requests.Response:
    """GET with retries that respect the NEON rate-limit headers."""
    for attempt in range(6):
        r = session.get(url, timeout=120, **kw)
        remaining = r.headers.get("x-ratelimit-remaining")  # absent on storage downloads
        if r.status_code == 429 or (remaining is not None and int(remaining) <= 1):
            time.sleep(int(r.headers.get("x-ratelimit-reset", 10)) + 1)
        if r.status_code == 200:
            return r
        if r.status_code in (403, 404):
            r.raise_for_status()
        time.sleep(2**attempt)
    r.raise_for_status()
    return r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--release", default="RELEASE-2026")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    s.headers["X-API-Token"] = os.environ["NEON_TOKEN"]
    product = get(s, f"{API}/products/{DPID}").json()["data"]

    # Gather file URLs per table for every site-month in the release.
    files = {t: {} for t in TABLES}
    for site in product["siteCodes"]:
        months = [
            m
            for rel in site["availableReleases"]
            if rel["release"] == a.release
            for m in rel["availableMonths"]
        ]
        for month in months:
            url = f"{API}/data/{DPID}/{site['siteCode']}/{month}"
            data = get(s, url, params={"release": a.release, "package": "basic"}).json()[
                "data"
            ]
            for f in data["files"]:
                table = next((t for t in TABLES if f".{t}." in f["name"]), None)
                if table is None or ".basic." not in f["name"]:
                    continue
                key = site["siteCode"] if table in SITE_LEVEL else f["name"]
                files[table][key] = f  # later months overwrite site-level copies
        print(site["siteCode"], len(months), "months", flush=True)

    for table, entries in files.items():
        with ThreadPoolExecutor(8) as pool:
            parts = list(
                pool.map(
                    lambda f: pd.read_csv(
                        io.BytesIO(get(s, f["url"]).content), dtype=str
                    ),
                    entries.values(),
                )
            )
        df = pd.concat(parts, ignore_index=True).drop_duplicates()
        df.to_csv(a.out_dir / f"{table}.csv", index=False)
        print(f"{table}: {len(entries)} files, {len(df):,} rows", flush=True)

    (a.out_dir / "RELEASE.txt").write_text(
        f"{DPID} {a.release} downloaded {time.strftime('%Y-%m-%d')}\n"
    )


if __name__ == "__main__":
    main()
