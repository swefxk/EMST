import argparse
import csv
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import ensure_dir, load_config, seed_everything


def load_truth_events(path):
    events = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            events.append(
                {
                    "event_id": int(row["event_id"]),
                    "t_start": int(row["t_start"]),
                    "t_end": int(row["t_end"]),
                    "t_center": float(row["t_center"]),
                    "geocell_id": int(row["geocell_id"]),
                    "band_id_true": int(row["band_id_true"]),
                    "power": float(row["power"]),
                    "bw": float(row["bw"]),
                }
            )
    return events


def load_geocells(path):
    geocells = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            geocells.append(
                {
                    "geocell_id": int(row["geocell_id"]),
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                }
            )
    return geocells


def generate_sensors(config, rng, num_geocell):
    s_cfg = config["data"]["sensor_reliability"]
    num_sensors = config["data"]["sensors"]
    sensors = []
    for sensor_id in range(num_sensors):
        geocell_id = int(rng.integers(0, num_geocell))
        reliability = float(rng.uniform(s_cfg["min"], s_cfg["max"]))
        sensors.append(
            {
                "sensor_id": sensor_id,
                "geocell_id": geocell_id,
                "reliability": reliability,
            }
        )
    return sensors


def save_sensors(out_dir, sensors):
    ensure_dir(out_dir)
    sensor_path = os.path.join(out_dir, "sensor.tsv")
    with open(sensor_path, "w", encoding="utf-8") as f:
        f.write("sensor_id\tgeocell_id\treliability\n")
        for row in sensors:
            f.write(
                f"{row['sensor_id']}\t{row['geocell_id']}\t"
                f"{row['reliability']:.6f}\n"
            )
    return sensor_path


def _distance(a, b, metric):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if metric == "euclidean":
        return float(np.sqrt(dx * dx + dy * dy))
    if metric == "chebyshev":
        return float(max(dx, dy))
    return float(dx + dy)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_observations(config, rng, truth_events, sensors, geocells):
    data_cfg = config["data"]
    obs_cfg = data_cfg["obs"]
    num_bands = data_cfg["bands"]
    time_steps = data_cfg["time_steps"]

    geocell_pos = {g["geocell_id"]: (g["x"], g["y"]) for g in geocells}
    sensor_pos = {}
    for sensor in sensors:
        sensor_pos[sensor["sensor_id"]] = geocell_pos.get(
            sensor["geocell_id"], (0, 0)
        )

    observations = []
    obs_id = 0
    true_count = 0
    false_count = 0

    fp_rel_weight = 0.15
    radius = obs_cfg.get("radius", 3)
    mean_observers = obs_cfg.get("mean_observers", 3.0)
    detect_a = obs_cfg.get("detect_a", 2.0)
    detect_b = obs_cfg.get("detect_b", 1.0)
    metric = obs_cfg.get("distance_metric", "manhattan")

    band_drift_max = int(obs_cfg.get("band_drift_max", 1))
    for event in truth_events:
        event_pos = geocell_pos.get(event["geocell_id"], (0, 0))
        candidates = []
        for sensor in sensors:
            s_pos = sensor_pos[sensor["sensor_id"]]
            dist = _distance(event_pos, s_pos, metric)
            if dist <= radius:
                candidates.append((sensor, dist))

        if not candidates:
            continue

        n_obs = int(rng.poisson(mean_observers))
        if n_obs <= 0:
            continue

        n_obs = min(n_obs, len(candidates))
        chosen = rng.choice(len(candidates), size=n_obs, replace=False)
        for idx in chosen:
            sensor, dist = candidates[idx]
            p_detect = (1.0 - obs_cfg["base_p_fn"]) * _sigmoid(
                detect_a - detect_b * dist
            )
            p_detect *= sensor["reliability"]
            p_detect = float(np.clip(p_detect, 0.0, 1.0))
            if rng.random() > p_detect:
                continue

            band_obs = event["band_id_true"]
            if rng.random() < obs_cfg["band_drift_prob"] and band_drift_max > 0:
                drift = int(rng.integers(-band_drift_max, band_drift_max + 1))
                if drift == 0:
                    drift = int(rng.choice([-1, 1]))
                band_obs = int(np.clip(band_obs + drift, 0, num_bands - 1))

            power_obs = event["power"] + rng.normal(0.0, obs_cfg["power_noise_std"])
            bw_obs = event["bw"] + rng.normal(0.0, obs_cfg["bw_noise_std"])
            conf = rng.beta(8.0, 2.0) * sensor["reliability"]
            conf = float(np.clip(conf, 0.0, 1.0))

            observations.append(
                {
                    "event_id": obs_id,
                    "t_center": event["t_center"],
                    "band_id_obs": band_obs,
                    "geocell_id_true": event["geocell_id"],
                    "sensor_id": sensor["sensor_id"],
                    "power_obs": power_obs,
                    "bw_obs": bw_obs,
                    "conf": conf,
                    "is_true": 1,
                    "band_id_true": event["band_id_true"],
                }
            )
            obs_id += 1
            true_count += 1

    for sensor in sensors:
        p_fp = obs_cfg["base_p_fp"] + (1.0 - sensor["reliability"]) * fp_rel_weight
        p_fp = float(np.clip(p_fp, 0.0, 0.5))
        for t in range(time_steps):
            if rng.random() < p_fp:
                band_obs = int(rng.integers(0, num_bands))
                power_obs = abs(rng.normal(0.0, obs_cfg["power_noise_std"]))
                bw_obs = abs(rng.normal(0.0, obs_cfg["bw_noise_std"]))
                conf = rng.beta(2.0, 8.0) * sensor["reliability"]
                conf = float(np.clip(conf, 0.0, 1.0))

                observations.append(
                    {
                        "event_id": obs_id,
                        "t_center": float(t),
                        "band_id_obs": band_obs,
                        "geocell_id_true": -1,
                        "sensor_id": sensor["sensor_id"],
                        "power_obs": power_obs,
                        "bw_obs": bw_obs,
                        "conf": conf,
                        "is_true": 0,
                        "band_id_true": -1,
                    }
                )
                obs_id += 1
                false_count += 1

    return observations, true_count, false_count


def save_observations(out_dir, observations):
    ensure_dir(out_dir)
    obs_path = os.path.join(out_dir, "observations.tsv")
    with open(obs_path, "w", encoding="utf-8") as f:
        f.write(
            "event_id\tt_center\tband_id_obs\tgeocell_id_true\tsensor_id\t"
            "power_obs\tbw_obs\tconf\tis_true\tband_id_true\n"
        )
        for row in observations:
            f.write(
                f"{row['event_id']}\t{row['t_center']:.3f}\t{row['band_id_obs']}\t"
                f"{row['geocell_id_true']}\t{row['sensor_id']}\t"
                f"{row['power_obs']:.6f}\t{row['bw_obs']:.6f}\t"
                f"{row['conf']:.6f}\t{row['is_true']}\t{row['band_id_true']}\n"
            )
    return obs_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--truth", required=True, help="Truth events TSV.")
    parser.add_argument("--geocell", required=True, help="Geocell TSV.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])
    rng = np.random.default_rng(config["seed"])

    truth_events = load_truth_events(args.truth)
    geocells = load_geocells(args.geocell)
    sensors = generate_sensors(config, rng, len(geocells))
    save_sensors(args.out, sensors)
    observations, true_count, false_count = generate_observations(
        config, rng, truth_events, sensors, geocells
    )
    save_observations(args.out, observations)

    print(f"Generated sensors: {len(sensors)}")
    print(f"Generated observations: {len(observations)}")
    print(f"  true: {true_count}  false: {false_count}")


if __name__ == "__main__":
    main()
