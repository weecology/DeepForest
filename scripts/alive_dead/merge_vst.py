"""Merge NEON VST mapping/tagging and apparent-individual tables, with stem coordinates.

Stem coordinates follow geoNEON::getLocTOS: the plot point location plus
``stemDistance`` along ``stemAzimuth``. Point locations come from the NEON TOS
plot points file. If a point has moved, the version that was valid on the
mapping date is used.

Outputs, written to ``--vst_dir``:
    vst_merged.csv       one row per apparent-individual record, with stem coordinates
    vst_individuals.csv  one row per individual, with its status history summarized
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEAD = ["Standing dead", "Dead, broken bole", "Downed", "Lost, presumed dead"]


def _norm_point(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def load_points(tos_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return current and versioned (historic) TOS point locations."""
    cur = pd.read_csv(next(tos_dir.glob("All_NEON_TOS_Plot_Points_V*.csv")))
    cur = cur.rename(columns={"horzUncert": "pointUncertainty"})
    cur = cur[["plotID", "pointID", "utmZone", "easting", "northing", "pointUncertainty"]]
    old = pd.read_csv(tos_dir / "TOS_versionedPoints.csv")
    old = old.rename(columns={"coordinateUncertainty": "pointUncertainty"})
    old = old[
        [
            "plotID",
            "pointID",
            "utmZone",
            "easting",
            "northing",
            "pointUncertainty",
            "versionStartDate",
            "versionEndDate",
        ]
    ]
    for d in (cur, old):
        d["pointID"] = _norm_point(d["pointID"])
    return cur, old


def locate_stems(mt: pd.DataFrame, cur: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Add stemEasting/stemNorthing to the mapping and tagging table."""
    mt = mt.copy()
    mt["pointID"] = _norm_point(mt["pointID"])
    mt["row"] = np.arange(len(mt))
    mt = mt.merge(cur, on=["plotID", "pointID"], how="left")

    # Override with the historic point version valid on the mapping date.
    day = pd.to_datetime(mt["date"], utc=True).dt.strftime("%Y%m%d").astype(int)
    hist = (
        mt[["row", "plotID", "pointID"]]
        .assign(day=day)
        .merge(old, on=["plotID", "pointID"])
    )
    hist = hist[(hist.day >= hist.versionStartDate) & (hist.day <= hist.versionEndDate)]
    hist = hist.drop_duplicates("row").set_index("row")
    cols = ["easting", "northing", "pointUncertainty", "utmZone"]
    mt = mt.set_index("row")
    mt.loc[hist.index, cols] = hist[cols]
    mt["pointVersioned"] = mt.index.isin(hist.index)

    az = np.deg2rad(mt["stemAzimuth"])
    mt["stemEasting"] = mt["easting"] + mt["stemDistance"] * np.sin(az)
    mt["stemNorthing"] = mt["northing"] + mt["stemDistance"] * np.cos(az)
    mt["utmZone"] = mt["utmZone"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    return mt.reset_index(drop=True).drop(columns=["easting", "northing"])


def summarize(merged: pd.DataFrame) -> pd.DataFrame:
    """One row per individual: latest status, last live date, first dead date."""
    m = merged.assign(year=merged["date"].dt.year)
    is_dead = m["plantStatus"].isin(DEAD)
    is_live = m["plantStatus"].str.startswith("Live", na=False)
    g = m.sort_values("date").groupby("individualID")
    out = g.agg(
        siteID=("siteID", "last"),
        plotID=("plotID", "last"),
        taxonID=("taxonID", "last"),
        stemEasting=("stemEasting", "last"),
        stemNorthing=("stemNorthing", "last"),
        utmZone=("utmZone", "last"),
        latestStatus=("plantStatus", "last"),
        latestDate=("date", "last"),
        latestHeight=("height", "last"),
        maxHeight=("height", "max"),
        latestCanopyPosition=("canopyPosition", "last"),
        nVisits=("date", "nunique"),
    )
    out["lastLiveYear"] = m[is_live].groupby("individualID")["year"].max()
    out["firstDeadYear"] = m[is_dead].groupby("individualID")["year"].min()
    out["latestDead"] = out["latestStatus"].isin(DEAD)
    return out.reset_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vst_dir", type=Path, required=True, help="Folder with stacked VST tables"
    )
    p.add_argument(
        "--tos_dir",
        type=Path,
        required=True,
        help="Unzipped All_NEON_TOS_Plots_V* folder",
    )
    a = p.parse_args()

    mt = pd.read_csv(a.vst_dir / "vst_mappingandtagging.csv", low_memory=False)
    ai = pd.read_csv(a.vst_dir / "vst_apparentindividual.csv", low_memory=False)
    cur, old = load_points(a.tos_dir)
    mt = locate_stems(mt, cur, old)

    # Use the most recent mapping record per individual (stems are occasionally re-mapped).
    mt["date"] = pd.to_datetime(mt["date"], utc=True)
    mt = mt.sort_values("date").drop_duplicates("individualID", keep="last")
    keep = [
        "individualID",
        "taxonID",
        "scientificName",
        "pointID",
        "stemDistance",
        "stemAzimuth",
        "stemEasting",
        "stemNorthing",
        "utmZone",
        "pointUncertainty",
        "pointVersioned",
    ]
    mt = mt[[c for c in keep if c in mt.columns]].rename(columns={"date": "mappingDate"})

    ai["date"] = pd.to_datetime(ai["date"], utc=True)
    merged = ai.merge(mt, on="individualID", how="left", validate="many_to_one")
    merged.to_csv(a.vst_dir / "vst_merged.csv", index=False)
    ind = summarize(merged)
    ind.to_csv(a.vst_dir / "vst_individuals.csv", index=False)

    print(
        f"apparentindividual rows {len(ai):,}; merged {len(merged):,}; "
        f"with stem coords {merged.stemEasting.notna().mean():.1%}"
    )
    print(f"individuals {len(ind):,}; latest status dead {ind.latestDead.sum():,}")


if __name__ == "__main__":
    main()
