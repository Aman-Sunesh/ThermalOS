from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thermalos.config import interventions_config
from thermalos.learning import update_intervention_priors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verified-effects", type=Path, required=True, help="CSV with intervention,observed_cooling_c")
    p.add_argument("--minimum-verified", type=int, default=3)
    args = p.parse_args()
    effects = pd.read_csv(args.verified_effects)
    result = update_intervention_priors(interventions_config(), effects, minimum_verified=args.minimum_verified)
    print(result.status)
    print(result.audit.to_string(index=False))
    print("No config file was modified; this is an auditable preview only.")


if __name__ == "__main__":
    main()
