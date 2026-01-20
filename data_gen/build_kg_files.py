import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_gen.generate_obs import (
    generate_observations,
    generate_sensors,
    save_observations,
    save_sensors,
)
from data_gen.generate_truth import (
    generate_bands,
    generate_geocells,
    generate_truth_events,
    save_truth,
)
from data_gen.split_by_time import save_split, split_by_time
from utils import ensure_dir, load_config, seed_everything


def save_events(out_dir, observations):
    event_path = os.path.join(out_dir, "event.tsv")
    with open(event_path, "w", encoding="utf-8") as f:
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
    return event_path


def build_entity_relation_dicts(out_dir, events, sensors, geocells, bands):
    entities = []
    for e in events:
        entities.append(f"event:{e['event_id']}")
    for s in sensors:
        entities.append(f"sensor:{s['sensor_id']}")
    for g in geocells:
        entities.append(f"geocell:{g['geocell_id']}")
    for b in bands:
        entities.append(f"band:{b['band_id']}")

    rels = ["observed_by", "located_in", "occurs_in", "overlaps_band"]

    ensure_dir(out_dir)
    entity_path = os.path.join(out_dir, "entity2id.txt")
    with open(entity_path, "w", encoding="utf-8") as f:
        f.write(f"{len(entities)}\n")
        for idx, ent in enumerate(entities):
            f.write(f"{ent}\t{idx}\n")

    rel_path = os.path.join(out_dir, "relation2id.txt")
    with open(rel_path, "w", encoding="utf-8") as f:
        f.write(f"{len(rels)}\n")
        for idx, rel in enumerate(rels):
            f.write(f"{rel}\t{idx}\n")
    return entity_path, rel_path


def build_triples(out_dir, events, sensors, split_ids):
    train_ids, valid_ids, test_ids = split_ids
    ensure_dir(out_dir)
    train_path = os.path.join(out_dir, "train.tsv")
    valid_path = os.path.join(out_dir, "valid.tsv")
    test_path = os.path.join(out_dir, "test.tsv")

    def write_triples(path, triple_rows):
        with open(path, "w", encoding="utf-8") as f:
            for h, r, t in triple_rows:
                f.write(f"{h}\t{r}\t{t}\n")

    train_rows = []
    valid_rows = []
    test_rows = []

    for event in events:
        h = f"event:{event['event_id']}"
        triples = [(h, "observed_by", f"sensor:{event['sensor_id']}")]
        if event["is_true"] == 1:
            triples.extend(
                [
                    (h, "occurs_in", f"geocell:{event['geocell_id_true']}"),
                    (h, "overlaps_band", f"band:{event['band_id_true']}"),
                ]
            )
        if event["event_id"] in train_ids:
            train_rows.extend(triples)
        elif event["event_id"] in valid_ids:
            valid_rows.extend(triples)
        else:
            test_rows.extend(triples)

    for sensor in sensors:
        triple = (
            f"sensor:{sensor['sensor_id']}",
            "located_in",
            f"geocell:{sensor['geocell_id']}",
        )
        train_rows.append(triple)

    write_triples(train_path, train_rows)
    write_triples(valid_path, valid_rows)
    write_triples(test_path, test_rows)
    return train_path, valid_path, test_path


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
    truth_events = generate_truth_events(config, rng)
    save_truth(args.out, geocells, bands, truth_events)

    sensors = generate_sensors(config, rng, len(geocells))
    save_sensors(args.out, sensors)

    observations, true_count, false_count = generate_observations(
        config, rng, truth_events, sensors, geocells
    )
    save_observations(args.out, observations)
    event_path = save_events(args.out, observations)

    train_ids, valid_ids, test_ids = split_by_time(
        [
            {"event_id": row["event_id"], "t_center": row["t_center"]}
            for row in observations
        ],
        config["splits"],
    )
    save_split(args.out, train_ids, valid_ids, test_ids)

    kg_dir = os.path.join(args.out, "kg")
    build_triples(kg_dir, observations, sensors, (train_ids, valid_ids, test_ids))
    build_entity_relation_dicts(kg_dir, observations, sensors, geocells, bands)

    print(f"Geocells: {len(geocells)} Bands: {len(bands)} Sensors: {len(sensors)}")
    print(f"Events: {len(observations)} (true {true_count}, false {false_count})")
    print(
        f"Split -> train {len(train_ids)} valid {len(valid_ids)} test {len(test_ids)}"
    )
    print(f"Event table: {event_path}")
    print(f"KG triples in {kg_dir}")


if __name__ == "__main__":
    main()
