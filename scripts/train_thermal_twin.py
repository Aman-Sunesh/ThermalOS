from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from thermalos.models.thermal_twin import ObservationalThermalTwin


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=["miami", "houston"], default="miami")
    args = p.parse_args()

    path = Path("data/processed") / f"{args.city}_tiles.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run build_city_features.py first: {path}")
    df = pd.read_csv(path)

    if "area" in df and df["area"].nunique() >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        tr, va = next(splitter.split(df, groups=df["area"]))
        train, valid = df.iloc[tr].copy(), df.iloc[va].copy()
    else:
        train, valid = train_test_split(df, test_size=0.25, random_state=42)

    model = ObservationalThermalTwin().fit(train)
    train_m = model.evaluate(train)
    valid_m = model.evaluate(valid)

    out = Path("models")
    out.mkdir(exist_ok=True)
    model_path = out / f"thermal_twin_{args.city}.joblib"
    model.save(model_path)
    metrics = {
        "city": args.city,
        "model_type": "observational_associational",
        "features": model.features_,
        "train": train_m.__dict__,
        "validation": valid_m.__dict__,
        "warning": "Metrics assess temperature association/prediction only; they do not establish causal intervention effects.",
    }
    (out / f"thermal_twin_{args.city}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Saved", model_path)


if __name__ == "__main__":
    main()
