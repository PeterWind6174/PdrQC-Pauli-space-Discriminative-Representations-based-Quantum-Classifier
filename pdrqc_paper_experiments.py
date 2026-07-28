from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pauli_progressive_selector import progressive_search, serializable_result
from pauli_upload_analysis import (
    X2,
    PreparedData,
    computational_basis_features,
    encode_states,
    group_qubit_wise_commuting,
    prepare_image_data,
    ring_edges,
)

ROOT = Path(__file__).resolve().parent
TORCH_REAL = torch.float64
TORCH_COMPLEX = torch.complex128


def _classifier(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2_000, random_state=seed)),
        ]
    )


def _fit_exact_baseline(
    name: str,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    test_features: np.ndarray,
    data: PreparedData,
    seed: int,
) -> dict[str, Any]:
    model = _classifier(seed)
    model.fit(
        np.vstack([train_features, validation_features]),
        np.concatenate([data.y_train, data.y_validation]),
    )
    prediction = model.predict(test_features)
    return {
        "method": name,
        "ideal_test_accuracy": float(accuracy_score(data.y_test, prediction)),
        "feature_count": int(test_features.shape[1]),
    }


def run_paper_readout_baselines(
    data: PreparedData,
    pdrqc_result: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    runtime = pdrqc_result["_runtime"]
    sequence = list(pdrqc_result["selected_sequence"])
    full_train, full_validation, full_test = runtime["feature_cache"][tuple(sequence)]

    pca_lr = _fit_exact_baseline(
        "pca_lr",
        data.X_train_pca,
        data.X_validation_pca,
        data.X_test_pca,
        data,
        seed,
    )
    full_pauli = _fit_exact_baseline(
        "full_pauli",
        full_train,
        full_validation,
        full_test,
        data,
        seed,
    )
    full_pauli["measurement_group_count"] = len(
        group_qubit_wise_commuting(runtime["pauli_strings"])
    )

    basis_train = computational_basis_features(data.X_train_angles, sequence)
    basis_validation = computational_basis_features(data.X_validation_angles, sequence)
    basis_test = computational_basis_features(data.X_test_angles, sequence)
    computational_basis = _fit_exact_baseline(
        "computational_basis",
        basis_train,
        basis_validation,
        basis_test,
        data,
        seed,
    )
    computational_basis["measurement_group_count"] = 1
    return [pca_lr, full_pauli, computational_basis]


def _ry_batch(theta: torch.Tensor) -> torch.Tensor:
    half = theta / 2.0
    gate = torch.zeros((theta.shape[0], 2, 2), dtype=TORCH_COMPLEX)
    gate[:, 0, 0] = torch.cos(half)
    gate[:, 1, 1] = torch.cos(half)
    gate[:, 0, 1] = -torch.sin(half)
    gate[:, 1, 0] = torch.sin(half)
    return gate


def _rz_batch(theta: torch.Tensor) -> torch.Tensor:
    half = theta / 2.0
    gate = torch.zeros((theta.shape[0], 2, 2), dtype=TORCH_COMPLEX)
    gate[:, 0, 0] = torch.exp(-1.0j * half)
    gate[:, 1, 1] = torch.exp(1.0j * half)
    return gate


def _apply_single_batch(
    states: torch.Tensor,
    gates: torch.Tensor,
    qubit: int,
    n_qubits: int,
) -> torch.Tensor:
    tensor = states.reshape(states.shape[0], *([2] * n_qubits))
    moved = torch.movedim(tensor, qubit + 1, -1)
    updated = torch.einsum("bij,b...j->b...i", gates, moved)
    return torch.movedim(updated, -1, qubit + 1).reshape(states.shape[0], -1)


def _apply_controlled_batch(
    states: torch.Tensor,
    gates: torch.Tensor,
    control: int,
    target: int,
    n_qubits: int,
) -> torch.Tensor:
    control_axis = control + 1
    target_axis = target + 1
    tensor = states.reshape(states.shape[0], *([2] * n_qubits))
    remaining = [
        axis
        for axis in range(1, n_qubits + 1)
        if axis not in (control_axis, target_axis)
    ]
    permutation = [0, *remaining, control_axis, target_axis]
    permuted = tensor.permute(permutation)
    flattened = permuted.reshape(states.shape[0], -1, 2, 2).clone()
    flattened[:, :, 1, :] = torch.einsum(
        "bij,bmj->bmi", gates, flattened[:, :, 1, :]
    )
    restored = flattened.reshape(permuted.shape)
    inverse = [0] * (n_qubits + 1)
    for index, original_axis in enumerate(permutation):
        inverse[original_axis] = index
    return restored.permute(inverse).reshape(states.shape[0], -1)


class DataReuploadingClassifier(torch.nn.Module):
    def __init__(self, n_qubits: int, n_classes: int, layers: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.layers = layers
        self.theta = torch.nn.Parameter(
            0.05 * torch.randn((layers, n_qubits, 2), dtype=TORCH_REAL)
        )

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        batch_size = angles.shape[0]
        states = torch.zeros(
            (batch_size, 2**self.n_qubits), dtype=TORCH_COMPLEX
        )
        states[:, 0] = 1.0
        x_gate = (
            torch.from_numpy(X2)
            .to(dtype=TORCH_COMPLEX)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        for layer in range(self.layers):
            for qubit in range(self.n_qubits):
                states = _apply_single_batch(
                    states, _ry_batch(angles[:, qubit]), qubit, self.n_qubits
                )
                states = _apply_single_batch(
                    states,
                    _ry_batch(self.theta[layer, qubit, 0].expand(batch_size)),
                    qubit,
                    self.n_qubits,
                )
                states = _apply_single_batch(
                    states,
                    _rz_batch(self.theta[layer, qubit, 1].expand(batch_size)),
                    qubit,
                    self.n_qubits,
                )
            for control, target in ring_edges(self.n_qubits):
                states = _apply_controlled_batch(
                    states, x_gate, control, target, self.n_qubits
                )

        probabilities = torch.abs(states) ** 2
        label_qubits = math.ceil(math.log2(self.n_classes))
        tensor = probabilities.reshape(batch_size, *([2] * self.n_qubits))
        if label_qubits < self.n_qubits:
            marginal = tensor.sum(
                dim=tuple(range(label_qubits + 1, self.n_qubits + 1))
            )
        else:
            marginal = tensor
        retained = marginal.reshape(batch_size, -1)[:, : self.n_classes]
        return retained / retained.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _train_drc(
    train_angles: np.ndarray,
    train_labels: np.ndarray,
    n_qubits: int,
    n_classes: int,
    epochs: int,
    seed: int,
) -> DataReuploadingClassifier:
    model = DataReuploadingClassifier(n_qubits, n_classes, layers=3, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    angles = torch.from_numpy(np.ascontiguousarray(train_angles)).to(TORCH_REAL)
    labels = torch.from_numpy(train_labels.astype(np.int64))
    for _ in range(epochs):
        optimizer.zero_grad()
        probabilities = model(angles)
        loss = -torch.log(
            probabilities[torch.arange(labels.shape[0]), labels].clamp_min(1e-12)
        ).mean()
        loss.backward()
        optimizer.step()
    return model


def _drc_probabilities(
    model: DataReuploadingClassifier,
    angles: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(angles)).to(TORCH_REAL)
        return model(tensor).cpu().numpy()


def _sample_probabilities(
    probabilities: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rows = [rng.multinomial(shots, row / row.sum()) / shots for row in probabilities]
    return np.asarray(rows, dtype=np.float64)


def run_data_reuploading(
    data: PreparedData,
    epochs: int,
    shot_budget: int,
    shot_repeats: int,
    seed: int,
) -> dict[str, Any]:
    train_angles = np.vstack([data.X_train_angles, data.X_validation_angles])
    train_labels = np.concatenate([data.y_train, data.y_validation])
    model = _train_drc(
        train_angles,
        train_labels,
        n_qubits=data.X_train_angles.shape[1],
        n_classes=len(data.class_ids),
        epochs=epochs,
        seed=seed,
    )
    probabilities = _drc_probabilities(model, data.X_test_angles)
    ideal_accuracy = float(
        accuracy_score(data.y_test, np.argmax(probabilities, axis=1))
    )
    finite: list[float] = []
    for repeat in range(shot_repeats):
        sampled = _sample_probabilities(
            probabilities,
            shots=shot_budget,
            rng=np.random.default_rng(seed + 30_013 * repeat),
        )
        finite.append(float(accuracy_score(data.y_test, np.argmax(sampled, axis=1))))
    n_qubits = data.X_train_angles.shape[1]
    return {
        "method": "data_reuploading",
        "ideal_test_accuracy": ideal_accuracy,
        "finite_shot_test_mean": float(np.mean(finite)),
        "finite_shot_test_std": float(np.std(finite)),
        "actual_shots_per_sample": shot_budget,
        "trainable_quantum_parameters": 6 * n_qubits,
        "layers": 3,
        "training_epochs": epochs,
    }


def _fidelity_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.square(np.abs(left @ np.conjugate(right.T)))


def run_quantum_kernel(
    data: PreparedData,
    shot_budget: int,
    shot_repeats: int,
    seed: int,
) -> dict[str, Any]:
    sequence = ["ry", "cz_ring", "ry", "cz_ring"]
    train_states = encode_states(data.X_train_angles, sequence)
    validation_states = encode_states(data.X_validation_angles, sequence)
    test_states = encode_states(data.X_test_angles, sequence)

    K_train = _fidelity_kernel(train_states, train_states)
    K_validation = _fidelity_kernel(validation_states, train_states)
    best_c = 1.0
    best_score = -np.inf
    for c_value in (0.1, 1.0, 10.0):
        model = SVC(C=c_value, kernel="precomputed", decision_function_shape="ovo")
        model.fit(K_train, data.y_train)
        score = accuracy_score(data.y_validation, model.predict(K_validation))
        if score > best_score:
            best_score = float(score)
            best_c = c_value

    train_validation_states = np.vstack([train_states, validation_states])
    train_validation_labels = np.concatenate([data.y_train, data.y_validation])
    K_final = _fidelity_kernel(train_validation_states, train_validation_states)
    K_test = _fidelity_kernel(test_states, train_validation_states)
    model = SVC(C=best_c, kernel="precomputed", decision_function_shape="ovo")
    model.fit(K_final, train_validation_labels)
    ideal_accuracy = float(accuracy_score(data.y_test, model.predict(K_test)))

    support = np.asarray(model.support_, dtype=np.int64)
    finite: list[float] = []
    for repeat in range(shot_repeats):
        rng = np.random.default_rng(seed + 40_009 * repeat)
        sampled_support = rng.binomial(
            shot_budget, np.clip(K_test[:, support], 0.0, 1.0)
        ) / float(shot_budget)
        noisy_kernel = np.zeros_like(K_test)
        noisy_kernel[:, support] = sampled_support
        finite.append(float(accuracy_score(data.y_test, model.predict(noisy_kernel))))

    n_qubits = data.X_train_angles.shape[1]
    return {
        "method": "quantum_kernel",
        "ideal_test_accuracy": ideal_accuracy,
        "finite_shot_test_mean": float(np.mean(finite)),
        "finite_shot_test_std": float(np.std(finite)),
        "support_vector_count": int(support.size),
        "shots_per_kernel_value": shot_budget,
        "actual_shots_per_sample": int(support.size * shot_budget),
        "selected_C": best_c,
        "feature_map_sequence": sequence,
        "resources_per_state_preparation": {
            "one_qubit_gates": 2 * n_qubits,
            "two_qubit_gates": 2 * n_qubits,
            "circuit_depth_proxy": 4 * n_qubits,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the classifiers and readouts reported in the PdrQC paper."
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
    parser.add_argument("--drc-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "pdrqc_tiny_paper_demo.json",
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
    pdrqc = progressive_search(
        data=data,
        max_pauli_weight=args.max_pauli_weight,
        max_rounds=args.max_rounds,
        target_accuracy=args.target_accuracy,
        shot_budget=args.shot_budget,
        shot_repeats=args.shot_repeats,
        score_eps=1e-3,
        seed=args.seed,
    )
    payload = {
        "scope": (
            "Small-sample execution check. It is not a reproduction of the full-data "
            "paper values."
        ),
        "configuration": {
            "dataset": args.dataset,
            "class_ids": class_ids,
            "n_qubits": args.n_qubits,
            "shot_budget": args.shot_budget,
            "shot_repeats": args.shot_repeats,
        },
        "pdrqc": serializable_result(pdrqc),
        "paper_readout_baselines": run_paper_readout_baselines(
            data, pdrqc_result=pdrqc, seed=args.seed
        ),
        "quantum_classifier_baselines": [
            run_data_reuploading(
                data,
                epochs=args.drc_epochs,
                shot_budget=args.shot_budget,
                shot_repeats=args.shot_repeats,
                seed=args.seed,
            ),
            run_quantum_kernel(
                data,
                shot_budget=args.shot_budget,
                shot_repeats=args.shot_repeats,
                seed=args.seed,
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
