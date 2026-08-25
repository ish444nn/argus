"""GraphSAGE: 64-d transaction embeddings and a secondary graph risk score.

Role in Argus
-------------
This model does **not** decide the analyst queue. XGBoost does. GraphSAGE
produces two things the investigation layer consumes as evidence:

A. a 64-dimensional embedding per transaction, written to pgvector, later
   used to find transactions with similar network behaviour among a reference
   pool of *historically labelled* cases;
B. an illicit probability, quoted in a case report as a second opinion from a
   neighbourhood-aware model.

Architecture
------------
Two SAGEConv layers, mean aggregation. Two because Elliptic's aggregated
features already summarise one hop, so a second hop is the first thing the
model can offer that the tabular features cannot; deeper stacks oversmooth on
a graph this sparse and buy nothing here.

    x -> SAGEConv(in, 128) -> ReLU -> Dropout
      -> SAGEConv(128, 64) -> ReLU        <- the embedding
      -> Linear(64, 1)                    <- the logit

Taking the embedding from the last message-passing layer rather than a
dedicated projection means it is exactly the representation the classifier
scores, so "similar embedding" really does mean "the model sees these two
transactions the same way".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

log = logging.getLogger(__name__)

EMBEDDING_DIM = 64
HIDDEN_DIM = 128


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = HIDDEN_DIM,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, embedding_dim, aggr="mean")
        self.head = nn.Linear(embedding_dim, 1)
        self.dropout = dropout
        self.embedding_dim = embedding_dim

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.relu(self.conv2(h, edge_index))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x, edge_index)).squeeze(-1)


@dataclass
class TrainingConfig:
    epochs: int = 200
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    hidden_dim: int = HIDDEN_DIM
    embedding_dim: int = EMBEDDING_DIM
    dropout: float = 0.3
    patience: int = 25
    seed: int = 42

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def positive_weight(graph: Data) -> torch.Tensor:
    """Ratio of licit to illicit among supervised nodes.

    Class weighting rather than resampling: duplicating or synthesising nodes
    on a graph would change its structure, which is the very thing the model
    is meant to learn from.
    """
    y = graph.y[graph.supervised_mask]
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    return torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)


@torch.no_grad()
def predict(model: GraphSAGE, graph: Data) -> tuple[np.ndarray, np.ndarray]:
    """Return (illicit probabilities, 64-d embeddings) for every node."""
    model.eval()
    embeddings = model.embed(graph.x, graph.edge_index)
    logits = model.head(embeddings).squeeze(-1)
    return torch.sigmoid(logits).numpy(), embeddings.numpy()


def train(
    train_graph: Data,
    val_graph: Data,
    config: TrainingConfig,
) -> tuple[GraphSAGE, dict]:
    """Fit on the training subgraph, early-stop on validation AUPRC.

    The validation graph is used only to *score* a candidate model. No
    gradient ever flows from it, and its labels never enter the loss.
    """
    from argus.ml.evaluate import ALERT_BUDGET, evaluate, threshold_for_budget

    set_seed(config.seed)
    model = GraphSAGE(
        in_channels=train_graph.x.shape[1],
        hidden_channels=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    )
    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pos_weight = positive_weight(train_graph)
    mask = train_graph.supervised_mask

    val_labels = np.where(
        val_graph.supervised_mask.numpy(),
        np.where(val_graph.y.numpy() > 0.5, "illicit", "licit"),
        "unknown",
    )

    best_auprc = -1.0
    best_state: dict | None = None
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimiser.zero_grad()
        logits = model(train_graph.x, train_graph.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[mask], train_graph.y[mask], pos_weight=pos_weight
        )
        loss.backward()
        optimiser.step()

        if epoch % 5 == 0 or epoch == config.epochs:
            scores, _ = predict(model, val_graph)
            metrics = evaluate(scores, val_labels, threshold_for_budget(scores, ALERT_BUDGET))
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss.detach().item(),
                    "val_auprc": metrics.auprc,
                }
            )
            if metrics.auprc > best_auprc:
                best_auprc = metrics.auprc
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= config.patience:
                log.info("early stop at epoch %d (best %d)", epoch, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log.info("best validation AUPRC %.4f at epoch %d", best_auprc, best_epoch)
    return model, {
        "best_epoch": best_epoch,
        "best_val_auprc": best_auprc,
        "history": history,
    }
