from pathlib import Path

from thermalos.data.miami import download_gtfs_stops


def main() -> None:
    stops = download_gtfs_stops()
    out = Path("data/raw/miami")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "gtfs_stops.csv"
    stops.to_csv(path, index=False)
    print(path, len(stops))


if __name__ == "__main__":
    main()
