"""HCA-GCN model and the manuscript-defined ablation variants."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def row_zscore(matrix, eps=1e-8):
    """Row-wise z-score used before the M5/M6 graph-fusion operations."""
    mean = matrix.mean(dim=-1, keepdim=True)
    std = matrix.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (matrix - mean) / std


class TimeSeriesEncoder(nn.Module):
    """Encode variable-duration ROI signals after deterministic truncation."""

    def __init__(self, roi_num, embed_dim, hidden_dim=64, num_heads=2, layers=2):
        super().__init__()
        self.roi_num = roi_num
        self.input_fc = nn.Linear(roi_num, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_fc = nn.Linear(hidden_dim, embed_dim)

    def forward(self, signals):
        # signals: [batch, ROI, time] -> Transformer tokens along time.
        if signals.size(1) != self.roi_num:
            raise ValueError(
                f"Expected {self.roi_num} ROIs, received {signals.size(1)}."
            )
        x = signals.transpose(1, 2)
        x = self.input_fc(x)
        x = self.encoder(x).mean(dim=1)
        x = self.output_fc(x)
        return x.unsqueeze(1).expand(-1, self.roi_num, -1)


class FCGenerator(nn.Module):
    """Generate a subject-specific associative graph from ROI embeddings."""

    def __init__(self, embed_dim, roi_num):
        super().__init__()
        self.roi_num = roi_num
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, roi_num * roi_num),
        )

    def forward(self, embeddings):
        batch_size = embeddings.size(0)
        x = embeddings.reshape(batch_size * self.roi_num, -1)
        x = self.mlp(x)
        x = x.view(
            batch_size, self.roi_num, self.roi_num, self.roi_num
        )
        return x.diagonal(dim1=2, dim2=3)


class CrossAttentionFusion(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, query_view, context_view):
        query = query_view.transpose(1, 2)
        context = context_view.transpose(1, 2)
        fused, _ = self.attention(query, context, context)
        return self.projection(fused.transpose(1, 2))


class CausalityEncoder(nn.Module):
    """Fuse nonlinear and linear causality-informed connectivity views."""

    def __init__(self, roi_num, embed_dim, num_heads=2):
        super().__init__()
        self.roi_num = roi_num
        self.cross_attention = CrossAttentionFusion(
            roi_num, embed_dim, num_heads
        )
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.projection = nn.Linear(32, embed_dim)

    def forward(self, nonlinear_gc, linear_gc):
        x = self.cross_attention(nonlinear_gc, linear_gc).unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, 32)
        x = self.projection(x)
        return F.interpolate(
            x.permute(0, 2, 1),
            size=self.roi_num,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)


class GraphPredictor(nn.Module):
    def __init__(self, node_feature_dim, roi_num, pool_ratio=0.5):
        super().__init__()
        self.roi_num = roi_num
        self.pool_ratio = pool_ratio
        self.gcn0 = nn.Linear(node_feature_dim, 128)
        self.gcn1 = nn.Linear(128, 64)
        self.gcn2 = nn.Linear(64, 32)
        self.bn0 = nn.BatchNorm1d(128)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.classifier = nn.Linear(32, 2)

    @staticmethod
    def _block(features, adjacency, layer, batch_norm):
        x = layer(features).transpose(1, 2)
        x = batch_norm(x).transpose(1, 2)
        x = F.leaky_relu(x)
        return torch.bmm(adjacency, x)

    def forward(self, adjacency, node_features):
        x = self._block(node_features, adjacency, self.gcn0, self.bn0)
        x = self._block(x, adjacency, self.gcn1, self.bn1)
        x = self.gcn2(x).transpose(1, 2)
        x = F.leaky_relu(self.bn2(x).transpose(1, 2))

        node_scores = torch.sigmoid(x.mean(dim=-1))
        keep = max(1, int(self.roi_num * self.pool_ratio))
        indices = node_scores.topk(keep, dim=1).indices
        x = torch.gather(
            x, 1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )
        return self.classifier(x.mean(dim=1)), node_scores


class HCAGCN(nn.Module):
    """Hierarchical multi-view brain connectivity fusion network.

    Supported variants:
      full: complete temporal + dual learned-graph model;
      m4_no_temporal: remove only the temporal branch;
      m5_single_fused_graph: row-standardize and combine FC/EC into one graph;
      m6_original_connectivity: use fixed original connectivity matrices.
    """

    VALID_VARIANTS = {
        "full",
        "m4_no_temporal",
        "m5_single_fused_graph",
        "m6_original_connectivity",
    }

    def __init__(
        self,
        model_config,
        roi_num=400,
        node_feature_dim=32,
        time_series=140,
    ):
        super().__init__()
        self.roi_num = int(roi_num)
        self.time_series = int(time_series)
        self.embed_dim = int(model_config.get("embedding_size", 32))
        if int(node_feature_dim) != self.embed_dim:
            raise ValueError(
                "node_feature_dim must equal embedding_size; the manuscript "
                "uses a 32-dimensional node representation."
            )

        self.variant = model_config.get("variant", "full").lower()
        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant {self.variant!r}; choose from "
                f"{sorted(self.VALID_VARIANTS)}."
            )

        heads = int(model_config.get("attention_heads", 2))
        hidden = int(model_config.get("temporal_hidden_dim", 64))
        layers = int(model_config.get("temporal_layers", 2))
        pool_ratio = float(model_config.get("pool_ratio", 0.5))
        self.single_graph_weight = float(
            model_config.get("single_graph_fc_weight", 0.5)
        )
        if not 0.0 <= self.single_graph_weight <= 1.0:
            raise ValueError("single_graph_fc_weight must be in [0, 1].")

        self.time_encoder = TimeSeriesEncoder(
            self.roi_num, self.embed_dim, hidden, heads, layers
        )
        self.association_fusion = CrossAttentionFusion(
            self.roi_num, self.embed_dim, heads
        )
        self.causality_encoder = CausalityEncoder(
            self.roi_num, self.embed_dim, heads
        )
        self.modality_fusion = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=heads,
            batch_first=True,
        )

        self.fc_generator = FCGenerator(self.embed_dim, self.roi_num)
        self.ec_generator = CrossAttentionFusion(
            self.roi_num, self.roi_num, heads
        )
        self.association_predictor = GraphPredictor(
            self.embed_dim, self.roi_num, pool_ratio
        )
        self.causality_predictor = GraphPredictor(
            self.embed_dim, self.roi_num, pool_ratio
        )
        self.single_predictor = GraphPredictor(
            self.embed_dim, self.roi_num, pool_ratio
        )
        self.logit_fusion = nn.Linear(4, 2)
        self.dropout = nn.Dropout(float(model_config.get("dropout", 0.5)))

    def _validate_shapes(self, signals, connectivity_views):
        expected_signals = (self.roi_num, self.time_series)
        if tuple(signals.shape[1:]) != expected_signals:
            raise ValueError(
                f"Time-series shape {tuple(signals.shape[1:])}; expected "
                f"{expected_signals}."
            )
        expected_matrix = (self.roi_num, self.roi_num)
        for name, matrix in connectivity_views:
            if tuple(matrix.shape[1:]) != expected_matrix:
                raise ValueError(
                    f"{name} shape {tuple(matrix.shape[1:])}; expected "
                    f"{expected_matrix}."
                )

    def _node_features(
        self, signals, nonlinear_gc, linear_gc, pearson, spearman
    ):
        embeddings = [
            self.association_fusion(pearson, spearman),
            self.causality_encoder(nonlinear_gc, linear_gc),
        ]
        if self.variant != "m4_no_temporal":
            embeddings.insert(0, self.time_encoder(signals))

        stacked = torch.stack(embeddings, dim=1)
        stacked = stacked.view(signals.size(0), -1, self.embed_dim)
        fused, _ = self.modality_fusion(stacked, stacked, stacked)
        return fused[:, : self.roi_num, :]

    def forward(
        self,
        final_fc,
        final_ngc,
        final_granger,
        final_pearson,
        final_spearman,
    ):
        views = [
            ("nonlinear_gc", final_ngc),
            ("linear_gc", final_granger),
            ("pearson", final_pearson),
            ("spearman", final_spearman),
        ]
        self._validate_shapes(final_fc, views)
        node_features = self._node_features(
            final_fc,
            final_ngc,
            final_granger,
            final_pearson,
            final_spearman,
        )

        if self.variant == "m6_original_connectivity":
            graph_fc = row_zscore((final_pearson + final_spearman) / 2.0)
            graph_ec = row_zscore((final_ngc + final_granger) / 2.0)
        else:
            graph_fc = self.fc_generator(node_features)
            graph_ec = self.ec_generator(final_ngc, final_granger)

        if self.variant == "m5_single_fused_graph":
            weight = self.single_graph_weight
            single_graph = (
                weight * row_zscore(graph_fc)
                + (1.0 - weight) * row_zscore(graph_ec)
            )
            logits, score = self.single_predictor(
                single_graph, node_features
            )
            return logits, graph_fc, graph_ec, score, score

        pred_fc, score_fc = self.association_predictor(
            graph_fc, node_features
        )
        pred_ec, score_ec = self.causality_predictor(
            graph_ec, node_features
        )
        logits = self.logit_fusion(
            self.dropout(torch.cat([pred_fc, pred_ec], dim=-1))
        )
        return logits, graph_fc, graph_ec, score_fc, score_ec
