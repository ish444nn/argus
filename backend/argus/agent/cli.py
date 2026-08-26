"""Command line entry points for the investigation layer.

python -m argus.agent.cli ingest-corpus   parse, embed and store the corpus
python -m argus.agent.cli corpus-status   what is stored, and by which model
python -m argus.agent.cli retrieve        run a retrieval query by pattern
python -m argus.agent.cli investigate     run the workflow for one case
"""

from __future__ import annotations

import argparse
import json
import logging

from argus.core.config import get_settings
from argus.core.logging import configure_logging

log = logging.getLogger("argus.agent")


def cmd_ingest_corpus(args: argparse.Namespace) -> None:
    from argus.agent import corpus
    from argus.db.session import SessionLocal

    with SessionLocal() as session:
        counts = corpus.ingest(session)
    print(json.dumps(counts, indent=2))


def cmd_corpus_status(_: argparse.Namespace) -> None:
    from argus.agent import corpus
    from argus.db.session import SessionLocal

    with SessionLocal() as session:
        print(json.dumps(corpus.corpus_status(session), indent=2))


def cmd_retrieve(args: argparse.Namespace) -> None:
    from argus.agent import retrieval
    from argus.db.session import SessionLocal

    query = retrieval.RetrievalQuery(
        text=args.query or " ".join(args.patterns), patterns=args.patterns
    )
    with SessionLocal() as session:
        result = retrieval.retrieve(session, query, k=args.k)

    print(f"patterns: {', '.join(result.patterns)}")
    print(f"query   : {result.query}\n")
    for chunk in result.chunks:
        print(f"  [{chunk.typology_id}] {chunk.title} -- {chunk.section_heading}")
        print(f"     similarity {chunk.similarity:.4f} | {chunk.citation()}")
        print(f"     {chunk.text.strip()[:160]}...\n")
    if not result.chunks:
        print("  (nothing retrieved)")


def cmd_investigate_top(args: argparse.Namespace) -> None:
    """Investigate the highest-scoring cases.

    Mostly for rebuilding a demo state: re-ingesting the corpus renumbers it
    and clears citations that no longer resolve, so the cases shown in the UI
    need investigating again afterwards. Running the test suite does exactly
    that, since its fixtures re-ingest.
    """
    from sqlalchemy import text

    from argus.db.session import SessionLocal
    from argus.services import investigation

    with SessionLocal() as session:
        ids = (
            session.execute(
                text("SELECT id FROM case_reports ORDER BY risk_score DESC, id LIMIT :n"),
                {"n": args.count},
            )
            .scalars()
            .all()
        )

    print(f"provider: {get_settings().llm_provider}; investigating {len(ids)} case(s)")
    for case_id in ids:
        with SessionLocal() as session:
            try:
                result = investigation.investigate(session, case_id)
                print(
                    f"  case {case_id}: {result['typology_assessment']} "
                    f"conf={result['confidence']} "
                    f"sources={len(result['retrieved_sources'])} "
                    f"fallback={result['used_fallback']}"
                )
            except Exception as exc:
                print(f"  case {case_id}: FAILED {type(exc).__name__}: {exc}")


def cmd_investigate(args: argparse.Namespace) -> None:
    from argus.agent.graph import investigate
    from argus.db.session import SessionLocal

    settings = get_settings()
    print(f"provider: {settings.llm_provider}")
    with SessionLocal() as session:
        state = investigate(session, args.case_id)

    if state.error:
        print(f"error: {state.error}")
        return

    narrative = state.generated.narrative
    print(f"\ncase {state.case_id} (tx {state.deterministic.tx_id})")
    print(f"  evidence items    : {len(state.deterministic.evidence)}")
    print(f"  retrieved sources : {[c.typology_id for c in state.retrieved.chunks]}")
    print(f"  provider/model    : {state.generated.provider}/{state.generated.model}")
    print(f"  attempts          : {state.generated.attempts}")
    print(f"  used fallback     : {state.generated.used_fallback}")
    print(f"  confidence        : {state.confidence} ({state.confidence_version})")
    if narrative:
        print(f"  typology          : {narrative.typology_assessment}")
        print(f"  recommended action: {narrative.recommended_action}")
        print(f"\n{narrative.summary}\n")
        for claim in narrative.claims:
            print(f"  - {claim.text}")
            print(f"      evidence={claim.evidence_ids} sources={claim.source_ids}")


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="argus.agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest-corpus", help="parse, embed and store the corpus").set_defaults(
        func=cmd_ingest_corpus
    )
    sub.add_parser("corpus-status", help="what is stored").set_defaults(func=cmd_corpus_status)

    retrieve_parser = sub.add_parser("retrieve", help="run a retrieval query")
    retrieve_parser.add_argument("patterns", nargs="+", help="pattern tags to filter on")
    retrieve_parser.add_argument("--query", default=None, help="query text")
    retrieve_parser.add_argument("--k", type=int, default=4)
    retrieve_parser.set_defaults(func=cmd_retrieve)

    investigate_parser = sub.add_parser("investigate", help="investigate one case")
    investigate_parser.add_argument("case_id", type=int)
    investigate_parser.set_defaults(func=cmd_investigate)

    top_parser = sub.add_parser("investigate-top", help="investigate the highest-scoring cases")
    top_parser.add_argument("--count", type=int, default=8)
    top_parser.set_defaults(func=cmd_investigate_top)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
