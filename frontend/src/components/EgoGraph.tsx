import { useQuery } from "@tanstack/react-query";
import { getNeighbourhood, type Neighbour } from "../api/client";
import { Note } from "./ui";

/**
 * The transaction's immediate neighbourhood, drawn from real edges.
 *
 * A deliberately small sketch rather than a force-directed graph: the useful
 * questions here are "how many counterparties, in which direction, and are any
 * of them already flagged", and a fixed layout answers those at a glance while
 * a physics simulation would move things around between visits.
 *
 * Senders sit left, recipients right, the subject in the middle — so the
 * direction of value is the horizontal axis and needs no arrowheads to read.
 * Flagged counterparties get a ring as well as a fill, so they are not
 * distinguished by colour alone.
 */

const W = 360;
const NODE_R = 5;

function column(items: Neighbour[], x: number, height: number) {
  const gap = height / (items.length + 1);
  return items.map((n, i) => ({ ...n, x, y: gap * (i + 1) }));
}

function nodeColour(n: Neighbour) {
  if (n.flagged) return "var(--risk-4)";
  if (n.risk_score !== null && n.risk_score >= 0.8) return "var(--risk-3)";
  return "var(--text-3)";
}

export function EgoGraph({ txId }: { txId: number }) {
  const { data, isPending, error } = useQuery({
    queryKey: ["neighbourhood", txId],
    queryFn: () => getNeighbourhood(txId),
  });

  if (isPending) return <div className="skeleton h-[110px]" aria-hidden />;
  if (error)
    return (
      <Note kind="error" title="Could not load the neighbourhood">
        {(error as Error).message}
      </Note>
    );
  if (!data || data.neighbours.length === 0)
    return (
      <Note title="No counterparties">
        This transaction has no recorded edges in the graph.
      </Note>
    );

  const incoming = data.neighbours.filter((n) => n.direction === "in");
  const outgoing = data.neighbours.filter((n) => n.direction === "out");
  const height = Math.max(72, Math.max(incoming.length, outgoing.length) * 22 + 36);

  const left = column(incoming, 46, height);
  const right = column(outgoing, W - 46, height);
  const cx = W / 2;
  const cy = height / 2;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${W} ${height}`}
        width="100%"
        height={height}
        role="img"
        aria-label={`${incoming.length} incoming and ${outgoing.length} outgoing counterparties`}
      >
        {[...left, ...right].map((n) => (
          <line
            key={`e-${n.tx_id}`}
            x1={n.x}
            y1={n.y}
            x2={cx}
            y2={cy}
            stroke={n.flagged ? "var(--risk-4)" : "var(--line-2)"}
            strokeWidth={n.flagged ? 1.2 : 1}
            opacity={n.flagged ? 0.7 : 0.5}
          />
        ))}

        {[...left, ...right].map((n) => (
          <g key={`n-${n.tx_id}`}>
            {n.flagged && (
              <circle
                cx={n.x}
                cy={n.y}
                r={NODE_R + 3}
                fill="none"
                stroke="var(--risk-4)"
                strokeWidth="1"
              />
            )}
            <circle cx={n.x} cy={n.y} r={NODE_R} fill={nodeColour(n)} />
            <title>
              {`tx ${n.tx_id} · ${n.direction === "in" ? "sends to" : "receives from"} this transaction`}
              {n.risk_score !== null ? ` · risk ${n.risk_score.toFixed(3)}` : ""}
              {n.flagged ? " · already flagged" : ""}
            </title>
          </g>
        ))}

        {/* The subject: a square, so it is not just a bigger dot. */}
        <rect
          x={cx - 8}
          y={cy - 8}
          width="16"
          height="16"
          fill="var(--surface-3)"
          stroke="var(--measured)"
          strokeWidth="1.5"
        />

        <text
          x={46}
          y={14}
          textAnchor="middle"
          className="eyebrow"
          fill="var(--text-3)"
          fontSize="9"
        >
          IN {incoming.length}
        </text>
        <text
          x={W - 46}
          y={14}
          textAnchor="middle"
          className="eyebrow"
          fill="var(--text-3)"
          fontSize="9"
        >
          OUT {outgoing.length}
        </text>
      </svg>

      <figcaption className="mt-2 text-[11px] text-[var(--text-3)]">
        Senders left, recipients right.{" "}
        {data.neighbours.some((n) => n.flagged) ? (
          <>
            Ringed counterparties are already in the queue.{" "}
          </>
        ) : null}
        {data.truncated && (
          <>
            Showing the {data.neighbours.length} highest-scoring of{" "}
            <span className="num">{data.total_degree}</span> edges.{" "}
          </>
        )}
        All counterparties share this transaction&rsquo;s batch — the dataset has no
        cross-batch edges.
      </figcaption>
    </figure>
  );
}
