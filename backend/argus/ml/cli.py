"""Command line entry points for the ML pipeline.

    python -m argus.ml.cli download    fetch the raw Elliptic files
    python -m argus.ml.cli inspect     report dataset shape and split sizes
    python -m argus.ml.cli ingest      load transactions and edges into Postgres
    python -m argus.ml.cli train       train every model, apply the promotion rule
    python -m argus.ml.cli embed       write GraphSAGE embeddings to pgvector

`train` and `embed` are separate because embedding writes to the database
while training does not, so training stays runnable with no database at all.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from argus.core.logging import configure_logging
from argus.ml import dataset, features, graph, registry, splits

DATA_DIR = Path(__file__).resolve().parents[2].parent / "data" / "elliptic"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPO_ROOT = Path(__file__).resolve().parents[2].parent
# MLflow put its plain-directory ("./mlruns") backend into maintenance mode and
# now refuses to open one. SQLite is the supported local equivalent: still a
# single file, still no server to run. `mlflow ui --backend-store-uri
# sqlite:///mlflow.db` reads it.
MLFLOW_TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"

log = logging.getLogger("argus.ml")


def _load() -> dataset.EllipticData:
    return dataset.load_cached(RAW_DIR, PROCESSED_DIR)


def cmd_download(_: argparse.Namespace) -> None:
    dataset.download(RAW_DIR)
    print(f"raw files ready in {RAW_DIR}")


def cmd_inspect(_: argparse.Namespace) -> None:
    data = _load()
    masks = splits.build_masks(data.timesteps)
    labels, counts = np.unique(data.labels, return_counts=True)

    report = {
        "nodes": data.n_nodes,
        "edges": data.n_edges,
        "features_stored": data.features.shape[1],
        "feature_variants": features.VARIANT_DIMS,
        "timestep_range": [int(data.timesteps.min()), int(data.timesteps.max())],
        "labels": dict(zip(labels.tolist(), counts.tolist(), strict=True)),
        "cross_timestep_edges": graph.count_cross_timestep_edges(data),
        "splits": {
            name: {
                "timesteps": splits.describe()[name],
                "nodes": int(masks[splits.Split(name)].sum()),
                "labelled": int((masks[splits.Split(name)] & (data.labels != "unknown")).sum()),
                "illicit": int((masks[splits.Split(name)] & (data.labels == "illicit")).sum()),
            }
            for name in splits.describe()
        },
    }
    print(json.dumps(report, indent=2))


def cmd_ingest(args: argparse.Namespace) -> None:
    from argus.db.session import SessionLocal
    from argus.ml import ingest as ingest_module

    data = _load()
    with SessionLocal() as session:
        counts = ingest_module.ingest(session, data, truncate=not args.append)
    print(json.dumps(counts, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    from argus.ml import train as train_module

    data = _load()
    results = train_module.run_all(data, tracking_uri=args.tracking_uri or MLFLOW_TRACKING_URI)

    rows = [*results["baselines"], results["graphsage"]]
    header = (
        f"{'model':<14}{'variant':<9}"
        f"{'val rec':>9}{'test rec':>10}{'test prec':>11}"
        f"{'test AUPRC':>12}{'alerts':>8}{'frozen rec':>12}"
    )
    print()
    print("  recall/precision at the 1% alert budget; 'frozen rec' applies the")
    print("  validation threshold verbatim, with no recalibration.")
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        budget = row.test.at_budget
        print(
            f"{row.name:<14}{row.feature_variant:<9}"
            f"{row.val.at_budget.recall:>9.3f}{budget.recall:>10.3f}"
            f"{budget.precision:>11.3f}{budget.auprc:>12.3f}"
            f"{budget.true_positives + budget.false_positives:>8d}"
            f"{row.test.at_fixed_threshold.recall:>12.3f}"
        )
    print()
    print(f"primary scorer: {results['primary']}")
    print(results["rationale"])


def cmd_embed(args: argparse.Namespace) -> None:
    import torch

    from argus.db.session import SessionLocal
    from argus.ml import gnn
    from argus.ml import ingest as ingest_module
    from argus.ml import train as train_module

    data = _load()
    masks = splits.build_masks(data.timesteps)
    metadata = registry.load_metadata("graphsage", args.models_root)

    model = gnn.GraphSAGE(
        in_channels=metadata.n_features,
        hidden_channels=metadata.hyperparameters["hidden_dim"],
        embedding_dim=metadata.hyperparameters["embedding_dim"],
        dropout=metadata.hyperparameters["dropout"],
    )
    model.load_state_dict(
        torch.load(
            registry.model_dir("graphsage", args.models_root) / "model.pt",
            weights_only=True,
        )
    )
    scaler = features.Scaler.from_dict(metadata.scaler)

    tx_ids, embeddings, scores = train_module.embed_all(
        data, masks, model, scaler, metadata.feature_variant
    )
    print(f"computed {len(tx_ids)} embeddings of dim {embeddings.shape[1]}")
    print(f"graph risk score range: {scores.min():.4f}..{scores.max():.4f}")

    with SessionLocal() as session:
        written = ingest_module.write_embeddings(
            session, tx_ids, embeddings, metadata.version, graph_scores=scores
        )
    print(f"wrote {written} embeddings to transaction_embeddings")


def cmd_export_demo(args: argparse.Namespace) -> None:
    """Write the seed SQL for the hosted database."""
    from argus.db.session import SessionLocal
    from argus.services import snapshot

    with SessionLocal() as session:
        if args.summary:
            print(json.dumps(snapshot.summarise(session, args.min_timestep), indent=2))
            return
        sql = snapshot.export(session, args.min_timestep, with_embeddings=args.with_embeddings)

    args.out.write_text(sql, encoding="utf-8")
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"wrote {args.out} ({size_mb:.1f} MB)")
    print('Apply with:  psql "$DATABASE_URL" -f ' + str(args.out))


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="argus.ml", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="fetch raw Elliptic files").set_defaults(func=cmd_download)
    sub.add_parser("inspect", help="report dataset shape and splits").set_defaults(func=cmd_inspect)

    ingest_parser = sub.add_parser("ingest", help="load into Postgres")
    ingest_parser.add_argument(
        "--append", action="store_true", help="do not truncate existing rows first"
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    train_parser = sub.add_parser("train", help="train and compare models")
    train_parser.add_argument("--tracking-uri", default=None)
    train_parser.set_defaults(func=cmd_train)

    export_parser = sub.add_parser("export-demo", help="write seed SQL for the hosted database")
    export_parser.add_argument("--out", type=Path, default=Path("demo-snapshot.sql"))
    export_parser.add_argument("--min-timestep", type=int, default=35)
    export_parser.add_argument(
        "--summary", action="store_true", help="report row counts without writing"
    )
    export_parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help=(
            "include transaction_embeddings (33 MB). Only useful for a "
            "deployment that hosts a Celery worker; the API never reads them."
        ),
    )
    export_parser.set_defaults(func=cmd_export_demo)

    embed_parser = sub.add_parser("embed", help="write embeddings to pgvector")
    embed_parser.add_argument("--models-root", type=Path, default=None)
    embed_parser.set_defaults(func=cmd_embed)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
