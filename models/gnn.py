from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool

from datatools import config
from physical_pass_model import (
    PHYSICAL_XPASS_LOGIT_ATTR,
    PHYSICAL_XPASS_VARIANTS,
    physical_xpass_model_variant,
    residual_distance_threshold,
    resolved_residual_clip_values,
)


class Encoder(nn.Module):
    def __init__(self, args: dict):
        super().__init__()

        self.args = args
        self.model_type = args.get("model", "gat")
        node_in_dim = args["node_in_dim"]
        edge_in_dim = args["edge_in_dim"]
        node_emb_dim = args["node_emb_dim"]
        graph_emb_dim = args["graph_emb_dim"]
        n_layers = args["gnn_layers"]
        n_heads = args["gnn_heads"]

        self.node_in_fc = nn.Sequential(nn.Linear(node_in_dim, node_emb_dim), nn.ReLU(inplace=True))

        self.gnn_layers = nn.ModuleList()
        self.node_layernorms = nn.ModuleList()
        self.activation = nn.ELU()

        for _ in range(n_layers):
            gnn_in_dim = node_in_dim + node_emb_dim if args["skip_conn"] else node_emb_dim
            if self.model_type == "gat":
                conv = GATConv(gnn_in_dim, node_emb_dim // n_heads, heads=n_heads, edge_dim=edge_in_dim)
            elif self.model_type == "gcn":
                conv = GCNConv(gnn_in_dim, node_emb_dim)
            elif self.model_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(gnn_in_dim, node_emb_dim),
                    nn.ReLU(),
                    nn.Linear(node_emb_dim, node_emb_dim),
                )
                conv = GINConv(mlp, train_eps=True)  # learnable epsilon adjusts self vs. neighbor aggregation
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            self.gnn_layers.append(conv)
            self.node_layernorms.append(nn.LayerNorm(node_emb_dim))

        self.node_out_fc = nn.Sequential(nn.Linear(node_emb_dim, node_emb_dim), nn.ReLU(inplace=True))
        self.node_layernorms.append(nn.LayerNorm(node_emb_dim))

        if graph_emb_dim > 0:
            self.graph_in_fc = nn.Sequential(nn.Linear(node_emb_dim, graph_emb_dim), nn.ReLU(inplace=True))
            self.graph_layernorm = nn.LayerNorm(graph_emb_dim)

    def forward(self, batch_graphs: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            batch_graphs: DataBatch with B graphs and total N nodes.
        Returns:
            node_embeddings: [N, z] tensor.
            graph_embeddings: [B, z] tensor.
        """
        node_embeddings = self.node_in_fc(batch_graphs.x)  # [N, z]
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.args["skip_conn"]:
                gnn_inputs = torch.cat([batch_graphs.x, node_embeddings], -1)  # [N, x+z]
            else:
                gnn_inputs = node_embeddings  # [N, z]

            if self.model_type == "gat":
                node_embeddings = gnn_layer(gnn_inputs, batch_graphs.edge_index, batch_graphs.edge_attr)  # [N, z]
            else:  # gcn or gin
                node_embeddings = gnn_layer(gnn_inputs, batch_graphs.edge_index)  # [N, z]
            node_embeddings = self.activation(self.node_layernorms[i](node_embeddings))

        if self.args["graph_emb_dim"] > 0:
            graph_embeddings = self.graph_layernorm(self.graph_in_fc(node_embeddings))  # [N, z]
            graph_embeddings = global_mean_pool(graph_embeddings, batch_graphs.batch)  # [B, z]
        else:
            graph_embeddings = None

        node_embeddings = self.node_layernorms[-1](self.node_out_fc(node_embeddings))
        return node_embeddings, graph_embeddings


class Decoder(nn.Module):
    def __init__(self, args: dict):
        super().__init__()

        self.args = args
        self.physical_variant = physical_xpass_model_variant(args)
        self.use_physical_xpass = (
            args.get("task") == "pass_success"
            and bool(args.get("use_physical_xpass", False))
            and self.physical_variant in PHYSICAL_XPASS_VARIANTS
        )
        self.physical_eps = float(args.get("physical_eps", 1e-4))
        self.residual_clip_value = args.get("residual_clip_value", None)
        self.residual_distance_threshold = residual_distance_threshold(args)
        self.short_residual_clip_value, self.long_residual_clip_value = resolved_residual_clip_values(args)
        self.latest_delta_gat = None
        node_in_dim = args["node_emb_dim"]
        graph_in_dim = args["graph_emb_dim"]
        mlp_dims = [args.get("mlp_h1_dim", 32), args.get("mlp_h2_dim", 8)]
        out_dim = args["out_dim"]

        dest_dim = 2
        if args.get("polar_dest", False):
            dest_dim += 2
        if args.get("more_dest_features", False):
            dest_dim += 5

        if args["skip_conn"]:
            node_in_dim += args["node_in_dim"]

        if "dest" in args["task"]:
            node_in_dim += dest_dim
            graph_in_dim += dest_dim

        if args["gnn_task"] == "node_selection":
            # {pass/action}_intent(_oppo_agn), {success/failure/dest}_receiver
            self.nodewise_mlp = nn.Sequential(
                nn.Linear(node_in_dim, mlp_dims[0]),
                nn.ReLU(),
                nn.Linear(mlp_dims[0], mlp_dims[1]),
                nn.ReLU(),
                nn.Linear(mlp_dims[1], 2),
                nn.GLU(),
            )
            if args["include_out"]:  # To estimate probabilities for the ball going out of bounds
                self.graphwise_mlp = nn.Sequential(
                    nn.Linear(graph_in_dim, mlp_dims[0]),
                    nn.ReLU(),
                    nn.Linear(mlp_dims[0], mlp_dims[1]),
                    nn.ReLU(),
                    nn.Linear(mlp_dims[1], 2),
                    nn.GLU(),
                )

        elif args["gnn_task"] in ["node_binary", "node_regression"]:
            # pass_success, outcome_{scoring/conceding/return}, intent_return(_oppo_agn)
            if self.use_physical_xpass and args.get("include_out", False):
                raise ValueError("include_out=True is not supported with physical xPass.")
            if self.use_physical_xpass and self.physical_variant == "gat_plus_phys_feature":
                node_in_dim += 1
            self.nodewise_mlp = nn.Sequential(
                nn.Linear(node_in_dim, mlp_dims[0]),
                nn.ReLU(),
                nn.Linear(mlp_dims[0], mlp_dims[1]),
                nn.ReLU(),
                nn.Linear(mlp_dims[1], out_dim),
            )
            if args["include_out"]:  # To estimate probabilities for the ball going out of bounds
                self.graphwise_mlp = nn.Sequential(
                    nn.Linear(graph_in_dim, mlp_dims[0]),
                    nn.ReLU(),
                    nn.Linear(mlp_dims[0], mlp_dims[1]),
                    nn.ReLU(),
                    nn.Linear(mlp_dims[1], out_dim),
                )
            if self.use_physical_xpass and self.physical_variant in {
                "gat_phys_logit_offset",
                "gat_phys_logit_offset_regularized",
            }:
                freeze_beta0 = bool(args.get("freeze_beta0", False))
                freeze_beta1 = args.get("freeze_beta1", None)
                if freeze_beta1 is None:
                    freeze_beta1 = not bool(args.get("learn_physical_scale", True))
                freeze_beta1 = bool(freeze_beta1)
                self.physical_beta0 = nn.Parameter(
                    torch.tensor(0.0, dtype=torch.float32),
                    requires_grad=not freeze_beta0,
                )
                self.physical_beta1 = nn.Parameter(
                    torch.tensor(1.0, dtype=torch.float32),
                    requires_grad=not freeze_beta1,
                )

        elif args["gnn_task"] in ["graph_binary", "graph_multiclass", "graph_regression"]:
            if args["task"].startswith("pass_dest"):
                self.graphwise_mlp = nn.Sequential(
                    nn.Linear(graph_in_dim, mlp_dims[0]),
                    nn.ReLU(),
                    nn.Linear(mlp_dims[0], mlp_dims[1]),
                    nn.ReLU(),
                    nn.Linear(mlp_dims[1], 2),
                    nn.GLU(),
                )
            else:
                # {overall/dest}_{scoring/conceding/return}, shot_blocking
                self.graphwise_mlp = nn.Sequential(
                    nn.Linear(graph_in_dim, mlp_dims[0]),
                    nn.ReLU(),
                    nn.Linear(mlp_dims[0], mlp_dims[1]),
                    nn.ReLU(),
                    nn.Linear(mlp_dims[1], out_dim),
                )

    def _clip_delta_gat(self, delta_gat: torch.Tensor, node_features: torch.Tensor | None) -> torch.Tensor:
        short_clip = self.short_residual_clip_value
        long_clip = self.long_residual_clip_value
        if short_clip is None and long_clip is None:
            return delta_gat
        if short_clip == long_clip:
            clip_value = float(short_clip)
            return clip_value * torch.tanh(delta_gat / clip_value)
        if node_features is None:
            raise ValueError("node_features are required for distance-specific residual clipping.")
        if node_features.shape[0] != delta_gat.shape[0] or node_features.shape[1] <= config.NODE_FEATURE_POSS_DIST:
            raise ValueError("node_features must include NODE_FEATURE_POSS_DIST for distance-specific residual clipping.")

        distances = node_features[:, config.NODE_FEATURE_POSS_DIST].to(device=delta_gat.device, dtype=delta_gat.dtype)
        short_mask = distances <= float(self.residual_distance_threshold)
        clipped = delta_gat
        if short_clip is not None:
            clip_value = float(short_clip)
            short_delta = clip_value * torch.tanh(delta_gat / clip_value)
            clipped = torch.where(short_mask, short_delta, clipped)
        if long_clip is not None:
            clip_value = float(long_clip)
            long_delta = clip_value * torch.tanh(delta_gat / clip_value)
            clipped = torch.where(~short_mask, long_delta, clipped)
        return clipped

    def forward(
        self,
        node_features: torch.Tensor = None,
        node_embeddings: torch.Tensor = None,
        graph_embeddings: torch.Tensor = None,
        batch_indices: torch.Tensor = None,
        batch_dests: torch.Tensor = None,
        physical_xpass_logit: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Arguments:
            node_features: [N, x] tensor.
            node_embeddings: [N, z] tensor.
            graph_embeddings: [B, z] tensor or None.
            batch_indices: [N] tensor indicating which batch each node belongs to.
            batch_dests: [B, d] tensor or None indicating ball directions or destinations.
        Returns:
            if gnn_task in ["node_selection", "node_binary", "node_regression"]:
                out: tensor with shape [N+B] if include_out else [N].
            elif gnn_task == ["graph_binary", "graph_regression"]:
                out: [B] tensor.
        """
        if "dest" in self.args["task"]:
            assert batch_dests is not None
        else:
            assert batch_dests is None

        if self.args["gnn_task"].startswith("node"):  # node_selection, node_binary, node_regression
            if self.args["skip_conn"]:
                node_inputs = torch.cat([node_features, node_embeddings], -1)  # [N, x+z]
            else:
                node_inputs = node_embeddings  # [N, z]

            if "dest" in self.args["task"]:
                assert batch_indices is not None
                nodewise_dests = batch_dests[batch_indices]  # [N, d]
                node_inputs = torch.cat([node_inputs, nodewise_dests], -1)  # [N, x+z+d] or [N, z+d]

            if self.use_physical_xpass:
                if physical_xpass_logit is None:
                    raise ValueError("physical_xpass_logit is required for this pass-success model variant.")
                physical_xpass_logit = physical_xpass_logit.to(device=node_inputs.device, dtype=node_inputs.dtype).reshape(-1)
                if physical_xpass_logit.shape[0] != node_inputs.shape[0]:
                    raise ValueError(
                        "physical_xpass_logit shape does not match node logits: "
                        f"{tuple(physical_xpass_logit.shape)} vs node_count={node_inputs.shape[0]}."
                    )
                if not torch.isfinite(physical_xpass_logit).all():
                    raise ValueError("physical_xpass_logit contains non-finite values.")
                if self.physical_variant == "gat_plus_phys_feature":
                    node_inputs = torch.cat([node_inputs, physical_xpass_logit.unsqueeze(-1)], -1)

            node_outputs = self.nodewise_mlp(node_inputs)  # [N, 1] or [N, 2]
            if node_outputs.shape[-1] == 1:
                node_outputs = node_outputs.squeeze(-1)  # [N]

            if self.use_physical_xpass and self.physical_variant in {
                "gat_phys_logit_offset",
                "gat_phys_logit_offset_regularized",
            }:
                delta_gat = node_outputs
                delta_gat = self._clip_delta_gat(delta_gat, node_features)
                self.latest_delta_gat = delta_gat
                node_outputs = self.physical_beta0 + self.physical_beta1 * physical_xpass_logit + delta_gat
            else:
                self.latest_delta_gat = node_outputs

            if not self.args["include_out"]:
                return node_outputs

            else:
                assert graph_embeddings is not None
                if "dest" in self.args["task"]:
                    residual_inputs = torch.cat([graph_embeddings, batch_dests], -1)  # [B, z+d]
                else:
                    assert batch_dests is None
                    residual_inputs = graph_embeddings  # [B, z]
                residual_outputs = self.graphwise_mlp(residual_inputs).squeeze(-1)  # [B]
                return torch.cat([node_outputs, residual_outputs], -1)  # [N+B]

        else:  # graph_binary, graph_multiclass, graph_regression
            assert graph_embeddings is not None

            if "dest" in self.args["task"]:
                graph_inputs = torch.cat([graph_embeddings, batch_dests], -1)  # [B, z+d]
                graph_outputs = self.graphwise_mlp(graph_inputs)  # [B, 1] or [B, 2]
                return graph_outputs.squeeze(-1) if graph_outputs.shape[-1] == 1 else graph_outputs  # [B] or [B, 2]

            else:  # scoring, conceding, shot_blocking
                return self.graphwise_mlp(graph_embeddings).squeeze(-1)  # [B] or [B, y]


class GNN(nn.Module):  # Graph-Encoder-Grid-Decoder
    def __init__(self, args: dict):
        super().__init__()

        self.args = args
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, batch_graphs: Batch, batch_dests: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            batch_graphs: DataBatch with B graphs and total N nodes.
            batch_dests: [B, d] tensor or None indicating ball directions or destinations.
        Returns:
            if gnn_task in ["node_selection", "node_binary"]:
                out: tensor with shape [N+B] if include_out else [N].
            elif gnn_task == "graph_binary":
                out: [B] tensor.
        """
        node_embeddings, graph_embeddings = self.encoder(batch_graphs)
        return self.decoder(
            batch_graphs.x,
            node_embeddings,
            graph_embeddings,
            batch_graphs.batch,
            batch_dests,
            getattr(batch_graphs, PHYSICAL_XPASS_LOGIT_ATTR, None),
        )

    def forward_grid(self, batch_graphs: Batch, grid_features: torch.Tensor) -> torch.Tensor:
        """
        Batched grid decoding: for each graph instance, score all grid cells at once.
        Args:
            batch_graphs: DataBatch with B graphs and total N nodes.
            grid_features: supports two shapes with grid size G = 105 x 68 = 7140 in general.
              - [G, d]: uniform grid features for all graphs.
              - [B, G, d]: per-graph grid features.
        Returns:
            out: [N, G] or [B, G] tensor of logits per (node, cell) or (graph, cell).
        """
        assert "dest" in self.args["task"]
        B = batch_graphs.num_graphs

        if grid_features.dim() == 3:  # [B, G, d]
            G = grid_features.size(1)
            grid_features = grid_features.reshape(B * G, -1)  # [B*G, d]
        else:
            assert grid_features.dim() == 2  # [G, d]
            G = grid_features.size(0)
            grid_features = grid_features.repeat(B, 1)  # [B*G, d]

        # Encode the input graphs once
        node_embeddings, graph_embeddings = self.encoder(batch_graphs)  # [N, z], [B, z]

        if self.args["gnn_task"] == "node_selection":
            node_feat_rep = batch_graphs.x.repeat_interleave(G, dim=0)  # [N*G, x]
            node_emb_rep = node_embeddings.repeat_interleave(G, dim=0)  # [N*G, z]
            graph_emb_rep = graph_embeddings.repeat_interleave(G, dim=0)  # [B*G, z]
            batch_idx_rep = batch_graphs.batch.repeat_interleave(G, dim=0)  # [N*G]

            logits = self.decoder(
                node_features=node_feat_rep,
                node_embeddings=node_emb_rep,
                graph_embeddings=graph_emb_rep,
                batch_indices=batch_idx_rep,
                batch_dests=grid_features,
            )  # [N*G] or [(N+B)*G]
            return logits.view(-1, G)  # [N, G] or [N+B, G]

        else:
            assert self.args["gnn_task"] in ["graph_binary", "graph_multiclass"]
            # Tile graph embeddings for every cell
            graph_emb_rep = graph_embeddings.repeat_interleave(G, dim=0)  # [B*G, z]

            # Decode for all (graph, cell) pairs
            logits = self.decoder(graph_embeddings=graph_emb_rep, batch_dests=grid_features)  # [B*G] or [B*G, 2]

            if logits.dim() == 1:  # [B*G]
                return logits.view(B, G)  # [B, G]
            else:  # [B*G, 2]
                return logits[:, 0].view(B, G), logits[:, 1].view(B, G)  # [B, G], [B, G]
