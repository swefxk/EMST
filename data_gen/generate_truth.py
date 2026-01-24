import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import ensure_dir, load_config, seed_everything


def generate_geocells(nx, ny):
    geocells = []
    geocell_id = 0
    for y in range(ny):
        for x in range(nx):
            geocells.append(
                {
                    "geocell_id": geocell_id,
                    "x": x,
                    "y": y,
                    "x_norm": x / max(1, nx - 1),
                    "y_norm": y / max(1, ny - 1),
                }
            )
            geocell_id += 1
    return geocells


def generate_bands(num_bands):
    bands = []
    for band_id in range(num_bands):
        bands.append(
            {
                "band_id": band_id,
                "f_center_norm": band_id / max(1, num_bands - 1),
            }
        )
    return bands


def generate_truth_events(config, rng):
    data_cfg = config["data"]
    nx = data_cfg["nx"]
    ny = data_cfg["ny"]
    num_geocell = nx * ny
    num_bands = data_cfg["bands"]
    t_steps = data_cfg["time_steps"]
    events_rate = data_cfg["events_per_step"]
    duration_range = data_cfg["event_duration"]

    power_mean = data_cfg["power"]["mean"]
    power_std = data_cfg["power"]["std"]
    bw_mean = data_cfg["bw"]["mean"]
    bw_std = data_cfg["bw"]["std"]

    events = []
    event_id = 0
    strong_cfg = data_cfg.get("strong_time", {})
    strong_enable = bool(strong_cfg.get("enable", False))
    prev_t = None
    prev_x = None
    prev_y = None
    prev_band = None
    geo_speed = float(strong_cfg.get("geo_speed", 0.4))
    geo_max_step = int(strong_cfg.get("geo_max_step", 3))
    band_drift_scale = float(strong_cfg.get("band_drift_scale", 0.3))
    band_max_step = int(strong_cfg.get("band_max_step", 3))
    for t in range(t_steps):
        if isinstance(events_rate, (list, tuple)):
            n_events = rng.integers(events_rate[0], events_rate[1] + 1)
        else:
            n_events = int(rng.poisson(float(events_rate)))
        for _ in range(n_events):
            duration = rng.integers(duration_range[0], duration_range[1] + 1)
            t_start = int(t)
            t_end = int(min(t_steps - 1, t_start + duration))
            t_center = (t_start + t_end) / 2.0
            if strong_enable and prev_t is not None:
                dt = max(1.0, t_start - prev_t)
                step = max(1, int(round(geo_speed * dt)))
                step = min(step, geo_max_step)
                dx = int(rng.integers(-step, step + 1))
                dy = int(rng.integers(-step, step + 1))
                x = int(np.clip(prev_x + dx, 0, nx - 1))
                y = int(np.clip(prev_y + dy, 0, ny - 1))
                geocell_id = y * nx + x

                drift = int(round(band_drift_scale * dt))
                drift = min(drift, band_max_step)
                if drift == 0:
                    band_id = prev_band
                else:
                    band_id = int((prev_band + rng.integers(-drift, drift + 1)) % num_bands)
            else:
                geocell_id = int(rng.integers(0, num_geocell))
                band_id = int(rng.integers(0, num_bands))
            power = max(0.1, rng.normal(power_mean, power_std))
            bw = max(0.1, rng.normal(bw_mean, bw_std))

            events.append(
                {
                    "event_id": event_id,
                    "t_start": t_start,
                    "t_end": t_end,
                    "t_center": t_center,
                    "geocell_id": geocell_id,
                    "band_id_true": band_id,
                    "power": power,
                    "bw": bw,
                }
            )
            prev_t = t_start
            prev_band = band_id
            prev_x = geocell_id % nx
            prev_y = geocell_id // nx
            event_id += 1
    return events


def save_truth(out_dir, geocells, bands, events):
    ensure_dir(out_dir)
    geocell_path = os.path.join(out_dir, "geocell.tsv")
    with open(geocell_path, "w", encoding="utf-8") as f:
        f.write("geocell_id\tx\ty\tx_norm\ty_norm\n")
        for row in geocells:
            f.write(
                f"{row['geocell_id']}\t{row['x']}\t{row['y']}\t"
                f"{row['x_norm']:.6f}\t{row['y_norm']:.6f}\n"
            )

    band_path = os.path.join(out_dir, "band.tsv")
    with open(band_path, "w", encoding="utf-8") as f:
        f.write("band_id\tf_center_norm\n")
        for row in bands:
            f.write(f"{row['band_id']}\t{row['f_center_norm']:.6f}\n")

    truth_path = os.path.join(out_dir, "truth_events.tsv")
    with open(truth_path, "w", encoding="utf-8") as f:
        f.write(
            "event_id\tt_start\tt_end\tt_center\tgeocell_id\tband_id_true\tpower\tbw\n"
        )
        for row in events:
            f.write(
                f"{row['event_id']}\t{row['t_start']}\t{row['t_end']}\t"
                f"{row['t_center']:.3f}\t{row['geocell_id']}\t"
                f"{row['band_id_true']}\t{row['power']:.6f}\t{row['bw']:.6f}\n"
            )
    return truth_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])
    rng = np.random.default_rng(config["seed"])

    geocells = generate_geocells(config["data"]["nx"], config["data"]["ny"])
    bands = generate_bands(config["data"]["bands"])
    events = generate_truth_events(config, rng)
    save_truth(args.out, geocells, bands, events)

    print(f"Generated geocells: {len(geocells)}")
    print(f"Generated bands: {len(bands)}")
    print(f"Generated truth events: {len(events)}")


if __name__ == "__main__":
    main()
