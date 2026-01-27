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


def generate_sources(config, rng):
    data_cfg = config["data"]
    src_cfg = data_cfg.get("source", {})
    num_sources = int(src_cfg.get("num_sources", 0))
    if num_sources <= 0:
        return []
    nx = data_cfg["nx"]
    ny = data_cfg["ny"]
    num_bands = data_cfg["bands"]
    motion_mode = src_cfg.get("motion_mode", "continuous")
    speed_min = float(src_cfg.get("speed_min", 0.2))
    speed_max = float(src_cfg.get("speed_max", 0.8))
    speed_max = max(speed_max, speed_min + 1e-6)
    sources = []
    for source_id in range(num_sources):
        if motion_mode == "discrete_step":
            directions = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
            dir_x, dir_y = directions[int(rng.integers(0, len(directions)))]
            speed = float(src_cfg.get("step_long", 1))
            vx = float(dir_x)
            vy = float(dir_y)
        else:
            angle = rng.uniform(0, 2 * np.pi)
            speed = rng.uniform(speed_min, speed_max)
            vx = float(speed * np.cos(angle))
            vy = float(speed * np.sin(angle))
            dir_x = float(np.sign(vx) or 1.0)
            dir_y = float(np.sign(vy) or 1.0)
        x_init = float(rng.uniform(0, nx - 1))
        y_init = float(rng.uniform(0, ny - 1))
        band_id = int(rng.integers(0, num_bands))
        sources.append(
            {
                "source_id": source_id,
                "x_init": x_init,
                "y_init": y_init,
                "vx": vx,
                "vy": vy,
                "vx_norm": vx / speed_max,
                "vy_norm": vy / speed_max,
                "speed": speed,
                "band_id": band_id,
                "dir_x": dir_x,
                "dir_y": dir_y,
            }
        )
    return sources


def _sample_dt(rng, cfg):
    short_max = int(cfg.get("dt_short_max", 3))
    long_prob = float(cfg.get("dt_long_prob", 0.3))
    long_min = int(cfg.get("dt_long_min", 6))
    long_max = int(cfg.get("dt_long_max", 18))
    if rng.random() < long_prob:
        return int(rng.integers(long_min, long_max + 1))
    return int(rng.integers(1, short_max + 1))


def _reflect(val, low, high):
    while val < low or val > high:
        if val < low:
            val = low + (low - val)
        if val > high:
            val = high - (val - high)
    return float(np.clip(val, low, high))


def generate_source_events(config, rng, sources):
    data_cfg = config["data"]
    src_cfg = data_cfg.get("source", {})
    nx = data_cfg["nx"]
    ny = data_cfg["ny"]
    num_bands = data_cfg["bands"]
    t_steps = int(data_cfg["time_steps"])
    duration_range = data_cfg["event_duration"]
    events_per_source = int(src_cfg.get("events_per_source", 120))
    start_frac = float(src_cfg.get("start_time_max_frac", 0.2))
    start_max = max(0, int(t_steps * start_frac))
    motion_mode = src_cfg.get("motion_mode", "continuous")
    vel_jitter = float(src_cfg.get("vel_jitter_std", 0.02))
    pos_noise = float(src_cfg.get("pos_noise_std", 0.2))
    boundary_mode = src_cfg.get("boundary_mode", "clip")
    step_short = int(src_cfg.get("step_short", 0))
    step_long = int(src_cfg.get("step_long", 8))
    band_drift_k = float(src_cfg.get("band_drift_k", 0.4))
    band_max_step = int(src_cfg.get("band_max_step", 3))

    power_mean = data_cfg["power"]["mean"]
    power_std = data_cfg["power"]["std"]
    bw_mean = data_cfg["bw"]["mean"]
    bw_std = data_cfg["bw"]["std"]

    events = []
    event_id = 0
    for source in sources:
        t = float(rng.integers(0, max(1, start_max + 1)))
        x = float(source["x_init"])
        y = float(source["y_init"])
        vx = float(source["vx"])
        vy = float(source["vy"])
        dir_x = float(source.get("dir_x", 1.0))
        dir_y = float(source.get("dir_y", 0.0))
        band_id = int(source["band_id"])
        for _ in range(events_per_source):
            dt = _sample_dt(rng, src_cfg)
            t += dt
            if t >= t_steps:
                break
            if motion_mode == "discrete_step":
                step = (
                    step_long
                    if dt > int(src_cfg.get("dt_short_max", 2))
                    else step_short
                )
                x = x + dir_x * step + rng.normal(0.0, pos_noise)
                y = y + dir_y * step + rng.normal(0.0, pos_noise)
            else:
                vx += rng.normal(0.0, vel_jitter)
                vy += rng.normal(0.0, vel_jitter)
                x = x + vx * dt + rng.normal(0.0, pos_noise)
                y = y + vy * dt + rng.normal(0.0, pos_noise)
            if boundary_mode == "reflect":
                x = _reflect(x, 0, nx - 1)
                y = _reflect(y, 0, ny - 1)
            else:
                x = float(np.clip(x, 0, nx - 1))
                y = float(np.clip(y, 0, ny - 1))

            drift = int(round(band_drift_k * dt + rng.normal(0.0, 0.5)))
            drift = int(np.clip(drift, -band_max_step, band_max_step))
            if drift != 0:
                band_id = int((band_id + drift) % num_bands)

            t_start = int(round(t))
            duration = rng.integers(duration_range[0], duration_range[1] + 1)
            t_end = int(min(t_steps - 1, t_start + duration))
            t_center = (t_start + t_end) / 2.0
            geocell_id = int(round(y)) * nx + int(round(x))
            power = max(0.1, rng.normal(power_mean, power_std))
            bw = max(0.1, rng.normal(bw_mean, bw_std))

            events.append(
                {
                    "event_id": event_id,
                    "source_id": source["source_id"],
                    "t_start": t_start,
                    "t_end": t_end,
                    "t_center": t_center,
                    "geocell_id": geocell_id,
                    "band_id_true": band_id,
                    "power": power,
                    "bw": bw,
                }
            )
            event_id += 1
    return events


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

    src_cfg = data_cfg.get("source", {})
    if bool(src_cfg.get("enable", False)):
        sources = generate_sources(config, rng)
        events = generate_source_events(config, rng, sources)
        return events, sources

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
                    band_id = int(
                        (prev_band + rng.integers(-drift, drift + 1)) % num_bands
                    )
            else:
                geocell_id = int(rng.integers(0, num_geocell))
                band_id = int(rng.integers(0, num_bands))
            power = max(0.1, rng.normal(power_mean, power_std))
            bw = max(0.1, rng.normal(bw_mean, bw_std))

            events.append(
                {
                    "event_id": event_id,
                    "source_id": -1,
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
    return events, []


def save_sources(out_dir, sources):
    if not sources:
        return None
    ensure_dir(out_dir)
    source_path = os.path.join(out_dir, "source.tsv")
    with open(source_path, "w", encoding="utf-8") as f:
        f.write("source_id\tx_init\ty_init\tvx_norm\tvy_norm\tspeed\n")
        for row in sources:
            f.write(
                f"{row['source_id']}\t{row['x_init']:.4f}\t{row['y_init']:.4f}\t"
                f"{row['vx_norm']:.6f}\t{row['vy_norm']:.6f}\t{row['speed']:.6f}\n"
            )
    return source_path


def save_truth(out_dir, geocells, bands, events, sources=None):
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
            "event_id\tsource_id\tt_start\tt_end\tt_center\tgeocell_id\tband_id_true\t"
            "power\tbw\n"
        )
        for row in events:
            f.write(
                f"{row['event_id']}\t{row.get('source_id', -1)}\t"
                f"{row['t_start']}\t{row['t_end']}\t"
                f"{row['t_center']:.3f}\t{row['geocell_id']}\t"
                f"{row['band_id_true']}\t{row['power']:.6f}\t{row['bw']:.6f}\n"
            )
    save_sources(out_dir, sources or [])
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
    events, sources = generate_truth_events(config, rng)
    save_truth(args.out, geocells, bands, events, sources)

    print(f"Generated geocells: {len(geocells)}")
    print(f"Generated bands: {len(bands)}")
    print(f"Generated truth events: {len(events)}")
    if sources:
        print(f"Generated sources: {len(sources)}")


if __name__ == "__main__":
    main()
