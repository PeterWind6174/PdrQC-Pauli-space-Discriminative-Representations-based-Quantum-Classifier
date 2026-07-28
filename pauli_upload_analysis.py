from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"

I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y2 = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
PAULI_SINGLE = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}

ATOMIC_BLOCKS = (
    "rx",
    "ry",
    "rz",
    "cz_ring",
    "cnot_ring",
    "crx_ring",
    "cry_ring",
    "crz_ring",
)


@dataclass(frozen=True)
class PreparedData:
    dataset: str
    class_ids: list[int]
    X_train_pca: np.ndarray
    X_validation_pca: np.ndarray
    X_test_pca: np.ndarray
    X_train_angles: np.ndarray
    X_validation_angles: np.ndarray
    X_test_angles: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    explained_variance_ratio: list[float]


def _balanced_rows(
    frame: pd.DataFrame,
    class_ids: list[int],
    per_class: int,
    seed: int,
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for class_id in class_ids:
        class_rows = frame[frame["label"].astype(int) == class_id]
        if len(class_rows) < per_class:
            raise ValueError(
                f"Class {class_id} has {len(class_rows)} rows, fewer than the requested {per_class}."
            )
        blocks.append(class_rows.sample(n=per_class, random_state=seed + class_id))
    return pd.concat(blocks, ignore_index=True)


def _labels_to_local(labels: np.ndarray, class_ids: list[int]) -> np.ndarray:
    mapping = {class_id: local_id for local_id, class_id in enumerate(class_ids)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int64)


def prepare_image_data(
    dataset: str,
    class_ids: list[int],
    n_qubits: int,
    train_per_class: int,
    validation_per_class: int,
    test_per_class: int,
    angle_scale: float = math.pi / 2.0,
    seed: int = 42,
) -> PreparedData:
    if dataset not in {"mnist", "fashion-mnist"}:
        raise ValueError("This paper package supports only MNIST and Fashion-MNIST.")
    if len(class_ids) < 2:
        raise ValueError("At least two class IDs are required.")
    if validation_per_class <= 0 or validation_per_class >= train_per_class:
        raise ValueError("validation_per_class must be between 1 and train_per_class - 1.")

    processed = DATASET_ROOT / dataset / "processed"
    train_path = processed / "train.csv.gz"
    test_path = processed / "test.csv.gz"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing processed data under {processed}. "
            "Run dataset/download_and_prepare_image_datasets.py first."
        )

    train_frame = _balanced_rows(
        pd.read_csv(train_path),
        class_ids=class_ids,
        per_class=train_per_class,
        seed=seed,
    )
    test_frame = _balanced_rows(
        pd.read_csv(test_path),
        class_ids=class_ids,
        per_class=test_per_class,
        seed=seed + 10_000,
    )

    search_blocks: list[pd.DataFrame] = []
    validation_blocks: list[pd.DataFrame] = []
    for class_id in class_ids:
        rows = train_frame[train_frame["label"].astype(int) == class_id].sample(
            frac=1.0, random_state=seed + 100 + class_id
        )
        validation_blocks.append(rows.iloc[:validation_per_class])
        search_blocks.append(rows.iloc[validation_per_class:])

    search_frame = pd.concat(search_blocks, ignore_index=True)
    validation_frame = pd.concat(validation_blocks, ignore_index=True)
    feature_columns = [column for column in train_frame.columns if column != "label"]

    X_train_raw = search_frame[feature_columns].to_numpy(dtype=np.float64)
    X_validation_raw = validation_frame[feature_columns].to_numpy(dtype=np.float64)
    X_test_raw = test_frame[feature_columns].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_validation_scaled = scaler.transform(X_validation_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    components = min(n_qubits, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])
    pca = PCA(n_components=components, svd_solver="full", random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_validation_pca = pca.transform(X_validation_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    if components < n_qubits:
        pad_width = n_qubits - components
        X_train_pca = np.pad(X_train_pca, ((0, 0), (0, pad_width)))
        X_validation_pca = np.pad(X_validation_pca, ((0, 0), (0, pad_width)))
        X_test_pca = np.pad(X_test_pca, ((0, 0), (0, pad_width)))

    feature_scale = np.max(np.abs(X_train_pca), axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0

    def to_angles(values: np.ndarray) -> np.ndarray:
        return np.clip(angle_scale * values / feature_scale, -math.pi, math.pi)

    return PreparedData(
        dataset=dataset,
        class_ids=class_ids,
        X_train_pca=X_train_pca,
        X_validation_pca=X_validation_pca,
        X_test_pca=X_test_pca,
        X_train_angles=to_angles(X_train_pca),
        X_validation_angles=to_angles(X_validation_pca),
        X_test_angles=to_angles(X_test_pca),
        y_train=_labels_to_local(search_frame["label"].to_numpy(), class_ids),
        y_validation=_labels_to_local(validation_frame["label"].to_numpy(), class_ids),
        y_test=_labels_to_local(test_frame["label"].to_numpy(), class_ids),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(float).tolist(),
    )


def rx(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [
            [math.cos(half), -1.0j * math.sin(half)],
            [-1.0j * math.sin(half), math.cos(half)],
        ],
        dtype=np.complex128,
    )


def ry(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [[math.cos(half), -math.sin(half)], [math.sin(half), math.cos(half)]],
        dtype=np.complex128,
    )


def rz(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [[np.exp(-1.0j * half), 0.0], [0.0, np.exp(1.0j * half)]],
        dtype=np.complex128,
    )


def _kron_many(operators: list[np.ndarray]) -> np.ndarray:
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def apply_single_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    operators = [I2] * n_qubits
    operators[qubit] = gate
    return _kron_many(operators) @ state


def apply_controlled_gate(
    state: np.ndarray,
    gate: np.ndarray,
    control: int,
    target: int,
    n_qubits: int,
) -> np.ndarray:
    if control == target:
        raise ValueError("control and target must be different")
    updated = state.copy()
    control_mask = 1 << (n_qubits - 1 - control)
    target_mask = 1 << (n_qubits - 1 - target)
    for basis_index in range(state.size):
        if (basis_index & control_mask) and not (basis_index & target_mask):
            paired_index = basis_index | target_mask
            amplitude_zero = state[basis_index]
            amplitude_one = state[paired_index]
            updated[basis_index] = (
                gate[0, 0] * amplitude_zero + gate[0, 1] * amplitude_one
            )
            updated[paired_index] = (
                gate[1, 0] * amplitude_zero + gate[1, 1] * amplitude_one
            )
    return updated


def ring_edges(n_qubits: int) -> list[tuple[int, int]]:
    if n_qubits < 2:
        return []
    if n_qubits == 2:
        return [(0, 1)]
    return [(qubit, qubit + 1) for qubit in range(n_qubits - 1)] + [
        (n_qubits - 1, 0)
    ]


def apply_atomic_block(
    state: np.ndarray,
    angles: np.ndarray,
    block: str,
) -> np.ndarray:
    n_qubits = int(angles.size)
    if block in {"rx", "ry", "rz"}:
        builder = {"rx": rx, "ry": ry, "rz": rz}[block]
        for qubit, angle in enumerate(angles):
            state = apply_single_qubit_gate(
                state, builder(float(angle)), qubit=qubit, n_qubits=n_qubits
            )
        return state
    if block in {"cz_ring", "cnot_ring"}:
        controlled_gate = Z2 if block == "cz_ring" else X2
        for control, target in ring_edges(n_qubits):
            state = apply_controlled_gate(
                state,
                controlled_gate,
                control=control,
                target=target,
                n_qubits=n_qubits,
            )
        return state
    if block in {"crx_ring", "cry_ring", "crz_ring"}:
        builder = {"crx_ring": rx, "cry_ring": ry, "crz_ring": rz}[block]
        for control, target in ring_edges(n_qubits):
            theta = 0.5 * float(angles[control] + angles[target])
            state = apply_controlled_gate(
                state,
                builder(theta),
                control=control,
                target=target,
                n_qubits=n_qubits,
            )
        return state
    raise ValueError(f"Unknown atomic block: {block}")


def encode_state(angles: np.ndarray, sequence: list[str]) -> np.ndarray:
    state = np.zeros(2 ** angles.size, dtype=np.complex128)
    state[0] = 1.0
    for block in sequence:
        state = apply_atomic_block(state, angles=angles, block=block)
    return state


def encode_states(angle_data: np.ndarray, sequence: list[str]) -> np.ndarray:
    return np.vstack([encode_state(row, sequence=sequence) for row in angle_data])


def generate_pauli_strings(n_qubits: int, max_weight: int) -> list[str]:
    strings: list[str] = []
    for weight in range(1, max_weight + 1):
        for positions in itertools.combinations(range(n_qubits), weight):
            for axes in itertools.product("XYZ", repeat=weight):
                label = ["I"] * n_qubits
                for position, axis in zip(positions, axes):
                    label[position] = axis
                strings.append("".join(label))
    return strings


def build_pauli_operators(pauli_strings: list[str]) -> np.ndarray:
    return np.stack(
        [_kron_many([PAULI_SINGLE[symbol] for symbol in label]) for label in pauli_strings]
    )


def pauli_features(
    angle_data: np.ndarray,
    sequence: list[str],
    pauli_operators: np.ndarray,
) -> np.ndarray:
    states = encode_states(angle_data, sequence=sequence)
    operated = np.matmul(pauli_operators, states.T)
    return np.real(np.einsum("nd,pdn->np", np.conjugate(states), operated))


def computational_basis_features(
    angle_data: np.ndarray,
    sequence: list[str],
) -> np.ndarray:
    states = encode_states(angle_data, sequence=sequence)
    return np.abs(states) ** 2


def qwc_compatible(left: str, right: str) -> bool:
    return all(a == "I" or b == "I" or a == b for a, b in zip(left, right))


def group_qubit_wise_commuting(pauli_strings: list[str]) -> list[list[int]]:
    groups: list[list[int]] = []
    for index, label in enumerate(pauli_strings):
        for group in groups:
            if all(qwc_compatible(label, pauli_strings[member]) for member in group):
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


def sample_pauli_features(
    exact_features: np.ndarray,
    groups: list[list[int]],
    total_shots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    if not groups:
        return exact_features.copy(), 0
    shots_per_group = max(1, total_shots // len(groups))
    sampled = np.empty_like(exact_features)
    for group in groups:
        probabilities = np.clip((exact_features[:, group] + 1.0) / 2.0, 0.0, 1.0)
        counts = rng.binomial(shots_per_group, probabilities)
        sampled[:, group] = 2.0 * counts / shots_per_group - 1.0
    return sampled, shots_per_group


def sequence_resources(sequence: list[str], n_qubits: int) -> dict[str, int]:
    one_qubit = sum(n_qubits for block in sequence if block in {"rx", "ry", "rz"})
    two_qubit = sum(
        n_qubits
        for block in sequence
        if block in {"cz_ring", "cnot_ring", "crx_ring", "cry_ring", "crz_ring"}
    )
    return {
        "sequence_length": len(sequence),
        "one_qubit_gates": one_qubit,
        "two_qubit_gates": two_qubit,
        "circuit_depth_proxy": one_qubit + two_qubit,
    }
