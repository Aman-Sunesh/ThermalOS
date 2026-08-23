from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from thermalos.api.fortyguard import FortyGuardClient
from thermalos.config import available_cities, city_config
from thermalos.data.fortyguard_parse import segmentation_to_row
from harvest_city import quantile_sample, temperature_spatial_sample


def _coord_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["area"].astype(str)
        + "::"
        + pd.to_numeric(df["lat"], errors="coerce").round(6).astype(str)
        + "::"
        + pd.to_numeric(df["lon"], errors="coerce").round(6).astype(str)
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Incrementally harvest sparse FortyGuard satellite morphology. "
            "Existing successful samples are preserved and deduplicated."
        )
    )
    p.add_argument("--city", choices=available_cities(), default="miami")
    p.add_argument("--samples-per-area", type=int, default=None)
    p.add_argument(
        "--sampling-strategy",
        choices=["temperature_spatial_maximin", "temperature_quantile"],
        default="temperature_spatial_maximin",
    )
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    load_dotenv()

    cfg = city_config(args.city)
    study_date = cfg["study_date"]
    granularity = int(cfg.get("granularity_m", 100))
    target_n = int(
        args.samples_per_area
        if args.samples_per_area is not None
        else cfg.get("harvest", {}).get("satellite_samples_per_area", 15)
    )

    source = Path("data/interim") / f"{cfg['city_key']}_fortyguard_tiles.csv"
    output = Path("data/interim") / f"{cfg['city_key']}_satellite_samples.csv"
    if not source.exists():
        raise FileNotFoundError(source)

    tiles = pd.read_csv(source)
    required = {"area", "lat", "lon", "temperature_c"}
    missing = required - set(tiles.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    existing = pd.DataFrame()
    if output.exists() and output.stat().st_size:
        try:
            existing = pd.read_csv(output)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
    if not existing.empty and {"area", "lat", "lon"}.issubset(existing.columns):
        existing = existing.drop_duplicates(subset=["area", "lat", "lon"], keep="last")

    rows = existing.to_dict(orient="records") if not existing.empty else []
    existing_keys = set(_coord_key(existing)) if not existing.empty else set()

    client = FortyGuardClient(poll_interval_s=5.0, task_timeout_s=900.0, verbose=True)

    print("==============================================")
    print("FORTYGUARD SATELLITE-ONLY HARVEST")
    print("==============================================")
    print("No TCM/exceedance/environment requests.")
    print("Sampling strategy:", args.sampling_strategy)
    print("Target samples per area:", target_n)
    print("Existing successful samples:", len(existing))

    # The active morphology file must follow one identical sampling protocol in
    # every city. Preserve all successful historical calls in a sidecar archive,
    # but expose exactly the deterministic target set (with fallback only if a
    # selected target call failed). This avoids giving one city denser morphology
    # merely because it had legacy samples from an earlier diagnostic harvest.
    target_keys: set[str] = set()

    for area_name, area_tiles in tiles.groupby("area", sort=False):
        valid = area_tiles.dropna(subset=["lat", "lon", "temperature_c"])
        if args.sampling_strategy == "temperature_quantile":
            sample = quantile_sample(valid, target_n)
            sample = sample.copy()
            sample["sampling_strategy"] = "temperature_quantile"
            sample["sampling_rank"] = np.arange(1, len(sample) + 1)
        else:
            sample = temperature_spatial_sample(valid, target_n)

        for _, target_row in sample.iterrows():
            target_keys.add(
                f"{area_name}::{round(float(target_row['lat']), 6)}::{round(float(target_row['lon']), 6)}"
            )

        current_area_count = 0 if existing.empty else int((existing["area"].astype(str) == str(area_name)).sum())
        print(f"\n=== {area_name} ===")
        print(f"Target selected locations: {len(sample)}; existing area samples: {current_area_count}")

        for sample_idx, (_, row) in enumerate(sample.iterrows(), start=1):
            key = f"{area_name}::{round(float(row['lat']), 6)}::{round(float(row['lon']), 6)}"
            if key in existing_keys:
                print(f"[{area_name}] satellite {sample_idx}/{len(sample)} already present; skip")
                continue

            print(f"\n[{area_name}] satellite {sample_idx}/{len(sample)}")
            try:
                result = client.satellite(
                    latitude=float(row["lat"]),
                    longitude=float(row["lon"]),
                    start_date=study_date,
                    start_time="14:00",
                    granularity=granularity,
                    refresh=args.refresh,
                )
                parsed = segmentation_to_row(result.result, "satellite")
                parsed["lat"] = float(row["lat"])
                parsed["lon"] = float(row["lon"])
                parsed["city"] = cfg["city_key"]
                parsed["area"] = area_name
                parsed["source_temperature_c"] = float(row["temperature_c"])
                parsed["sampling_strategy"] = str(row.get("sampling_strategy", args.sampling_strategy))
                parsed["sampling_rank"] = int(row.get("sampling_rank", sample_idx))
                rows.append(parsed)
                existing_keys.add(key)

                current = pd.DataFrame(rows)
                if {"area", "lat", "lon"}.issubset(current.columns):
                    current = current.drop_duplicates(subset=["area", "lat", "lon"], keep="last")
                current.to_csv(output, index=False)
                print("  saved; total unique successful satellite samples =", len(current))
            except Exception as exc:
                print("  SATELLITE FAILED:", exc)

    final = pd.DataFrame(rows)
    archive_path = output.with_name(output.stem + "_archive.csv")
    if not final.empty and {"area", "lat", "lon"}.issubset(final.columns):
        archive = final.drop_duplicates(subset=["area", "lat", "lon"], keep="last").copy()
        archive.to_csv(archive_path, index=False)

        keyed = archive.copy()
        keyed["_coord_key"] = _coord_key(keyed)
        active = keyed[keyed["_coord_key"].isin(target_keys)].copy()

        # If a newly selected target failed, retain deterministic successful
        # legacy samples as fallback so an AOI does not fall below target_n.
        # Successful selected targets always take precedence.
        for area_name in tiles["area"].astype(str).drop_duplicates().tolist():
            have = int((active["area"].astype(str) == area_name).sum())
            if have >= target_n:
                continue
            active_keys = set(active["_coord_key"].astype(str))
            extras = keyed[
                (keyed["area"].astype(str) == area_name)
                & (~keyed["_coord_key"].astype(str).isin(active_keys))
            ].copy()
            if extras.empty:
                continue
            if "sampling_rank" in extras.columns:
                extras["_fallback_rank"] = pd.to_numeric(extras["sampling_rank"], errors="coerce").fillna(1e9)
            else:
                extras["_fallback_rank"] = 1e9
            extras = extras.sort_values(["_fallback_rank", "lat", "lon"], kind="stable")
            active = pd.concat([active, extras.head(target_n - have)], ignore_index=True)

        final = (
            active.drop(columns=[c for c in ["_coord_key", "_fallback_rank"] if c in active.columns])
            .drop_duplicates(subset=["area", "lat", "lon"], keep="last")
        )
        final.to_csv(output, index=False)

    print("\n==============================================")
    print("SATELLITE HARVEST COMPLETE")
    print("==============================================")
    print("Active protocol samples:", len(final))
    if not final.empty:
        print("Per area:", final["area"].value_counts().to_dict())
    if archive_path.exists():
        print("Historical successful-sample archive:", archive_path)
    print("Saved active protocol set:", output)


if __name__ == "__main__":
    main()
