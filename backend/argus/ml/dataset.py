"""Loading and parsing the raw Elliptic dataset.

File layout, verified by inspecting the actual files (not assumed):

`elliptic_txs_features.csv` -- **no header row**, 203,769 rows x 167 columns:
    col 0   txId
    col 1   time step, an integer 1..49
    col 2..94   93 further "local" transaction features
    col 95..166  72 "aggregated" features (one-hop neighbour aggregates,
                 computed by the dataset authors)

The literature's "166 features" counts the time step as the first local
feature: 1 + 93 + 72 = 166. We never feed the time step to a model
(a model that keys on it cannot extrapolate to future time steps), so the
feature matrix this module returns holds **165** columns and the time step is
carried separately. The first 93 of those are the local group.

`elliptic_txs_classes.csv` -- header `txId,class`. Values are `1` (illicit,
4,545), `2` (licit, 42,019) and the literal string `unknown` (157,205).

`elliptic_txs_edgelist.csv` -- header `txId1,txId2`, 234,355 directed
transaction -> transaction edges. Verified: no duplicates, no self-loops, and
every edge references a transaction present in the feature file.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from argus.db.enums import Label

log = logging.getLogger(__name__)

# The dataset is redistributed by the PyTorch Geometric maintainers, which is
# also where our torch-geometric dependency fetches it from. Kaggle, the
# original host, requires an account, which would make ingestion
# non-reproducible for anyone cloning this repository.
SOURCE_URL = "https://data.pyg.org/datasets/elliptic"
FILE_NAMES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)

# Column counts in the raw feature file.
RAW_COLUMNS = 167
N_FEATURES = 165  # 166 in the literature, minus the time step
N_LOCAL_FEATURES = 93  # 94 in the literature, minus the time step
N_AGGREGATED_FEATURES = N_FEATURES - N_LOCAL_FEATURES  # 72

MIN_TIMESTEP = 1
MAX_TIMESTEP = 49

# Raw class codes as they appear in the CSV.
_CLASS_TO_LABEL = {"1": Label.ILLICIT, "2": Label.LICIT, "unknown": Label.UNKNOWN}


@dataclass(frozen=True)
class EllipticData:
    """The parsed dataset.

    All arrays are aligned by row index, and `tx_ids[i]` is the transaction at
    index `i`. `edge_index` holds those same indices, not raw transaction ids,
    so it can be handed straight to PyTorch Geometric.
    """

    tx_ids: np.ndarray  # int64  (n_nodes,)
    timesteps: np.ndarray  # int16  (n_nodes,)
    labels: np.ndarray  # <U8    (n_nodes,) -- illicit / licit / unknown
    features: np.ndarray  # float32 (n_nodes, 165)
    edge_index: np.ndarray  # int64  (2, n_edges) -- row indices

    @property
    def n_nodes(self) -> int:
        return len(self.tx_ids)

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    def index_of(self, tx_id: int) -> int:
        """Row index for a raw transaction id."""
        (matches,) = np.nonzero(self.tx_ids == tx_id)
        if not len(matches):
            raise KeyError(f"unknown transaction id {tx_id}")
        return int(matches[0])


def download(raw_dir: Path) -> None:
    """Fetch and unzip any missing raw file. Idempotent."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILE_NAMES:
        target = raw_dir / name
        if target.exists():
            log.info("%s already present", name)
            continue
        archive = raw_dir / f"{name}.zip"
        if not archive.exists():
            log.info("downloading %s", name)
            with urlopen(f"{SOURCE_URL}/{name}.zip") as response:
                archive.write_bytes(response.read())
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
        log.info("extracted %s", name)


def load(raw_dir: Path) -> EllipticData:
    """Parse the three raw CSVs into aligned arrays.

    Raises if the files do not have the shape this module documents, so a
    changed or truncated download fails loudly instead of silently training on
    the wrong columns.
    """
    features_path = raw_dir / "elliptic_txs_features.csv"
    classes_path = raw_dir / "elliptic_txs_classes.csv"
    edges_path = raw_dir / "elliptic_txs_edgelist.csv"

    for path in (features_path, classes_path, edges_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run `argus-ml download` first")

    # header=None: the feature file genuinely has no header row. Reading it
    # with the default would silently consume the first transaction.
    raw = pd.read_csv(features_path, header=None, dtype=np.float64)
    if raw.shape[1] != RAW_COLUMNS:
        raise ValueError(f"expected {RAW_COLUMNS} feature columns, found {raw.shape[1]}")

    tx_ids = raw.iloc[:, 0].to_numpy(dtype=np.int64)
    timesteps = raw.iloc[:, 1].to_numpy(dtype=np.int16)
    features = np.ascontiguousarray(raw.iloc[:, 2:].to_numpy(dtype=np.float32))
    del raw

    if features.shape[1] != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} model features, found {features.shape[1]}")
    if len(np.unique(tx_ids)) != len(tx_ids):
        raise ValueError("duplicate transaction ids in the feature file")
    if timesteps.min() < MIN_TIMESTEP or timesteps.max() > MAX_TIMESTEP:
        raise ValueError(f"time steps outside {MIN_TIMESTEP}..{MAX_TIMESTEP}")

    classes = pd.read_csv(classes_path, dtype={"txId": np.int64, "class": str})
    unexpected = set(classes["class"].unique()) - set(_CLASS_TO_LABEL)
    if unexpected:
        raise ValueError(f"unexpected class codes: {sorted(unexpected)}")

    # Align labels to feature row order via the transaction id.
    label_by_tx = dict(zip(classes["txId"], classes["class"], strict=True))
    if set(label_by_tx) != set(tx_ids.tolist()):
        raise ValueError("feature and class files cover different transactions")
    labels = np.array(
        [_CLASS_TO_LABEL[label_by_tx[int(tx)]].value for tx in tx_ids],
        dtype="<U8",
    )

    index_by_tx = {int(tx): i for i, tx in enumerate(tx_ids)}
    edges = pd.read_csv(edges_path, dtype=np.int64)
    src = edges["txId1"].map(index_by_tx)
    dst = edges["txId2"].map(index_by_tx)
    if src.isna().any() or dst.isna().any():
        raise ValueError("edge list references transactions absent from the feature file")
    edge_index = np.vstack([src.to_numpy(dtype=np.int64), dst.to_numpy(dtype=np.int64)])

    log.info(
        "loaded %d transactions, %d edges, %d features",
        len(tx_ids),
        edge_index.shape[1],
        features.shape[1],
    )
    return EllipticData(
        tx_ids=tx_ids,
        timesteps=timesteps,
        labels=labels,
        features=features,
        edge_index=edge_index,
    )


def cache_path(processed_dir: Path) -> Path:
    return processed_dir / "elliptic.npz"


def load_cached(raw_dir: Path, processed_dir: Path) -> EllipticData:
    """Parse once, then reuse a compact .npz on later runs.

    Parsing the 690 MB feature CSV takes about half a minute; every training
    run paying that is a waste. The cache is a pure function of the raw files.
    """
    path = cache_path(processed_dir)
    if path.exists():
        with np.load(path, allow_pickle=False) as blob:
            return EllipticData(
                tx_ids=blob["tx_ids"],
                timesteps=blob["timesteps"],
                labels=blob["labels"],
                features=blob["features"],
                edge_index=blob["edge_index"],
            )

    data = load(raw_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        tx_ids=data.tx_ids,
        timesteps=data.timesteps,
        labels=data.labels,
        features=data.features,
        edge_index=data.edge_index,
    )
    log.info("cached parsed dataset at %s", path)
    return data
