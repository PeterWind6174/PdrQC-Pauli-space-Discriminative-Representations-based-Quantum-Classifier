from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pauli_upload_analysis import (
    ATOMIC_BLOCKS,
    PreparedData,
    build_pauli_operators,
    generate_pauli_strings,
    group_qubit_wise_commuting,
    pauli_features,
    prepare_image_data,
    sample_pauli_features,
    sequence_resources,
)

ROOT = Path(__file__).resolve().parent


def build_classifier(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2_000, random_state=seed)),
        ]
    )


def multiclass_pauli_scores(
    features: np.ndarray,
    labels: np.ndarray,
    eps: float,
) -> np.ndarray:
    classes = np.unique(labels)
    global_mean = features.mean(axis=0)
    between = np.zeros(features.shape[1], dtype=np.float64)
    within = np.zeros(features.shape[1], dtype=np.float64)
    for class_id in classes:
        class_features = features[labels == class_id]
        class_mean = class_features.mean(axis=0)
        between += class_features.shape[0] * np.square(class_mean - global_mean)
        within += np.square(class_features - class_mean).sum(axis=0)
    sample_count = max(1, features.shape[0])
    return (between / sample_count) / (within / sample_count + eps)


def pauli_budget(k: int, total_count: int) -> int:
    requested = 28 if k == 2 else 4 * k + 20
    return min(requested, total_count)


def _fit_accuracy(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
) -> tuple[Pipeline, float]:
    classifier = build_classifier(seed)
    classifier.fit(train_features, y_train)
    prediction = classifier.predict(validation_features)
    return classifier, float(accuracy_score(y_validation, prediction))


def evaluate_candidate(
    sequence: list[str],
    data: PreparedData,
    pauli_strings: list[str],
    pauli_operators: np.ndarray,
    feature_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray, np.ndarray]],
    shot_budget: int,
    shot_repeats: int,
    score_eps: float,
    seed: int,
) -> dict[str, Any]:
    key = tuple(sequence)
    if key not in feature_cache:
        feature_cache[key] = (
            pauli_features(data.X_train_angles, sequence, pauli_operators),
            pauli_features(data.X_validation_angles, sequence, pauli_operators),
            pauli_features(data.X_test_angles, sequence, pauli_operators),
        )
    train_full, validation_full, test_full = feature_cache[key]

    scores = multiclass_pauli_scores(train_full, data.y_train, eps=score_eps)
    budget = pauli_budget(k=len(data.class_ids), total_count=len(pauli_strings))
    selected_indices = np.argsort(scores)[::-1][:budget]
    selected_strings = [pauli_strings[int(index)] for index in selected_indices]

    train_selected = train_full[:, selected_indices]
    validation_selected = validation_full[:, selected_indices]
    classifier, ideal_validation_accuracy = _fit_accuracy(
        train_selected,
        validation_selected,
        data.y_train,
        data.y_validation,
        seed=seed,
    )

    groups = group_qubit_wise_commuting(selected_strings)
    shot_accuracies: list[float] = []
    shots_per_group = 0
    for repeat in range(shot_repeats):
        sampled_validation, shots_per_group = sample_pauli_features(
            validation_selected,
            groups=groups,
            total_shots=shot_budget,
            rng=np.random.default_rng(seed + 10_003 * repeat + len(sequence)),
        )
        prediction = classifier.predict(sampled_validation)
        shot_accuracies.append(float(accuracy_score(data.y_validation, prediction)))

    finite_mean = float(np.mean(shot_accuracies))
    finite_std = float(np.std(shot_accuracies))
    ranking_key = (
        ideal_validation_accuracy,
        finite_mean,
        -finite_std,
    )

    return {
        "sequence": list(sequence),
        "selected_indices": selected_indices,
        "selected_strings": selected_strings,
        "groups": groups,
        "scores": scores,
        "ranking_key": ranking_key,
        "ideal_validation_accuracy": ideal_validation_accuracy,
        "finite_shot_validation_mean": finite_mean,
        "finite_shot_validation_std": finite_std,
        "shots_per_group": shots_per_group,
        "train_features": train_selected,
        "validation_features": validation_selected,
        "test_features": test_full[:, selected_indices],
    }


def progressive_search(
    data: PreparedData,
    max_pauli_weight: int,
    max_rounds: int,
    target_accuracy: float,
    shot_budget: int,
    shot_repeats: int,
    score_eps: float,
    seed: int,
) -> dict[str, Any]:
    n_qubits = data.X_train_angles.shape[1]
    pauli_strings = generate_pauli_strings(n_qubits, max_weight=max_pauli_weight)
    pauli_operators = build_pauli_operators(pauli_strings)
    feature_cache: dict[
        tuple[str, ...], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}

    previous_round_best: list[str] = []
    best_overall: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = []

    for round_index in range(1, max_rounds + 1):
        sequences = (
            [[block] for block in ATOMIC_BLOCKS]
            if round_index == 1
            else [previous_round_best + [block] for block in ATOMIC_BLOCKS]
        )
        candidates = [
            evaluate_candidate(
                sequence=sequence,
                data=data,
                pauli_strings=pauli_strings,
                pauli_operators=pauli_operators,
                feature_cache=feature_cache,
                shot_budget=shot_budget,
                shot_repeats=shot_repeats,
                score_eps=score_eps,
                seed=seed + round_index,
            )
            for sequence in sequences
        ]
        round_best = max(candidates, key=lambda item: item["ranking_key"])
        previous_round_best = list(round_best["sequence"])
        if best_overall is None or round_best["ranking_key"] > best_overall["ranking_key"]:
            best_overall = round_best

        trace.append(
            {
                "round": round_index,
                "best_sequence": round_best["sequence"],
                "ideal_validation_accuracy": round_best[
                    "ideal_validation_accuracy"
                ],
                "finite_shot_validation_mean": round_best[
                    "finite_shot_validation_mean"
                ],
                "finite_shot_validation_std": round_best[
                    "finite_shot_validation_std"
                ],
            }
        )
        if round_best["ideal_validation_accuracy"] >= target_accuracy:
            break

    if best_overall is None:
        raise RuntimeError("No upload sequence was evaluated.")

    final_train = np.vstack(
        [best_overall["train_features"], best_overall["validation_features"]]
    )
    final_labels = np.concatenate([data.y_train, data.y_validation])
    classifier = build_classifier(seed)
    classifier.fit(final_train, final_labels)
    ideal_prediction = classifier.predict(best_overall["test_features"])
    ideal_test_accuracy = float(accuracy_score(data.y_test, ideal_prediction))

    finite_test_accuracies: list[float] = []
    shots_per_group = 0
    for repeat in range(shot_repeats):
        sampled_test, shots_per_group = sample_pauli_features(
            best_overall["test_features"],
            groups=best_overall["groups"],
            total_shots=shot_budget,
            rng=np.random.default_rng(seed + 1_000_003 + 20_011 * repeat),
        )
        finite_prediction = classifier.predict(sampled_test)
        finite_test_accuracies.append(
            float(accuracy_score(data.y_test, finite_prediction))
        )

    selected_strings = list(best_overall["selected_strings"])
    return {
        "method": "pdrqc",
        "dataset": data.dataset,
        "class_ids": data.class_ids,
        "n_qubits": n_qubits,
        "max_pauli_weight": max_pauli_weight,
        "candidate_pauli_count": len(pauli_strings),
        "selected_pauli_count": len(selected_strings),
        "selected_pauli_strings": selected_strings,
        "measurement_groups": [
            [selected_strings[index] for index in group]
            for group in best_overall["groups"]
        ],
        "measurement_group_count": len(best_overall["groups"]),
        "shot_budget": shot_budget,
        "shots_per_group": shots_per_group,
        "actual_shots_per_sample": shots_per_group
        * len(best_overall["groups"]),
        "selected_sequence": best_overall["sequence"],
        "ideal_validation_accuracy": best_overall["ideal_validation_accuracy"],
        "finite_shot_validation_mean": best_overall[
            "finite_shot_validation_mean"
        ],
        "finite_shot_validation_std": best_overall[
            "finite_shot_validation_std"
        ],
        "ideal_test_accuracy": ideal_test_accuracy,
        "finite_shot_test_mean": float(np.mean(finite_test_accuracies)),
        "finite_shot_test_std": float(np.std(finite_test_accuracies)),
        "search_trace": trace,
        "resources": sequence_resources(best_overall["sequence"], n_qubits),
        "preprocessing": {
            "pca_explained_variance_ratio": data.explained_variance_ratio,
            "train_samples": int(data.y_train.size),
            "validation_samples": int(data.y_validation.size),
            "test_samples": int(data.y_test.size),
        },
        "_runtime": {
            "classifier": classifier,
            "test_features": best_overall["test_features"],
            "pauli_strings": pauli_strings,
            "pauli_operators": pauli_operators,
            "feature_cache": feature_cache,
        },
    }


def serializable_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "_runtime"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the compact PdrQC implementation on MNIST or Fashion-MNIST."
    )
    parser.add_argument("--dataset", choices=("mnist", "fashion-mnist"), default="mnist")
    parser.add_argument("--classes", default="0,1,2")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--max-pauli-weight", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--validation-per-class", type=int, default=5)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--shot-budget", type=int, default=2048)
    parser.add_argument("--shot-repeats", type=int, default=5)
    parser.add_argument("--score-eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "pdrqc_tiny_mnist.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_ids = [int(value) for value in args.classes.split(",") if value.strip()]
    data = prepare_image_data(
        dataset=args.dataset,
        class_ids=class_ids,
        n_qubits=args.n_qubits,
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )
    result = progressive_search(
        data=data,
        max_pauli_weight=args.max_pauli_weight,
        max_rounds=args.max_rounds,
        target_accuracy=args.target_accuracy,
        shot_budget=args.shot_budget,
        shot_repeats=args.shot_repeats,
        score_eps=args.score_eps,
        seed=args.seed,
    )
    payload = serializable_result(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
