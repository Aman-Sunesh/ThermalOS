from dotenv import load_dotenv

from thermalos.api.fortyguard import FortyGuardClient
from thermalos.config import city_config
from thermalos.geo import square_aoi


def main() -> None:
    load_dotenv()
    cfg = city_config("miami")
    area_name, area = next(iter(cfg["areas"].items()))
    client = FortyGuardClient(
        poll_interval_s=5.0,
        task_timeout_s=300.0,
        request_timeout_s=45.0,
        verbose=True,
    )
    aoi = square_aoi(area["lat"], area["lon"], 1.0)
    print(f"Testing {area_name}, Miami-Dade...", flush=True)
    result = client.heatmap(
        polygon_aoi=aoi,
        start_date=cfg["study_date"],
        start_time="14:00",
        filter_type=1,
        granularity=100,
        refresh=True,
    )
    n = len(result.result.get("map_data", {}).get("features", []))
    print(
        f"OK: activity={result.activity_id} tiles={n} "
        f"cache={result.from_cache} elapsed={result.elapsed_s:.1f}s"
    )


if __name__ == "__main__":
    main()
