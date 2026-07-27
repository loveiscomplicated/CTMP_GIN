import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GINConv

from src.models.entity_embedding import EntityEmbeddingBatch3

def _make_gin_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )

class GIN(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 col_info, 
                 gin_dim, 
                 gin_layer_num, 
                 num_classes,
                 train_eps=True,
                 use_los=True,
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        original_col_list = list(col_info[0])
        original_col_dims = list(col_info[1])
        self.full_col_list = original_col_list
        self.full_col_dims = original_col_dims
        self.feature_col_indices = [
            int(idx) for idx, name in enumerate(original_col_list) if str(name) != "LOS"
        ]
        self._feature_index_by_original = {
            original_idx: feature_idx
            for feature_idx, original_idx in enumerate(self.feature_col_indices)
        }
        self.col_list = [original_col_list[idx] for idx in self.feature_col_indices]
        self.col_dims = [original_col_dims[idx] for idx in self.feature_col_indices]
        self.discharge_col_index = [
            self._feature_index_by_original[int(idx)]
            for idx, name in enumerate(original_col_list)
            if str(name).endswith("_D") and int(idx) in self._feature_index_by_original
        ]
        self.use_los = use_los
        self.max_los = int(kwargs.get("max_los", 37))
        # self.col_dims.append(self.max_los + 1) # LOS needs to be included in GIN, as it's excluded in col_info. --> remove_LOS parameter makes LOS included in col_info
        self.gin_dim = gin_dim
        self.gin_layer_num = gin_layer_num
        self.train_eps = train_eps
        self.num_classes = num_classes
        self.node_feature_dim = embedding_dim

        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, 
                                                            embedding_dim=embedding_dim)
        self.los_embedding_layer = nn.Embedding(self.max_los + 1, self.node_feature_dim)
        
        self.gin_layers = nn.ModuleList()

        gin_layer1 = GINConv(nn=_make_gin_mlp(self.node_feature_dim, gin_dim), eps=0, train_eps=self.train_eps)
        self.gin_layers.append(gin_layer1)
        
        for _ in range(self.gin_layer_num - 1):
            gin_layer_hidden = GINConv(nn=_make_gin_mlp(gin_dim, gin_dim), eps=0, train_eps=self.train_eps)
            self.gin_layers.append(gin_layer_hidden)

        # 분류기 레이어 정의
        out_dim = 1 if self.num_classes == 2 else self.num_classes
        self.classifier_dim = self.gin_dim * self.gin_layer_num
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_dim, self.classifier_dim * 2),
            nn.ReLU(),
            nn.Linear(self.classifier_dim * 2, out_dim)
        )
        self._edge_format_cache = None

    def _validate_x_feature_width(self, x: torch.Tensor) -> None:
        expected_width = len(self.col_dims)
        if x.shape[1] != expected_width:
            raise ValueError(
                "GIN x feature width mismatch: "
                f"expected {expected_width} non-LOS feature columns, got {x.shape[1]}. "
                "LOS must be passed through the separate los argument."
            )

    def encode_los(self, los: torch.Tensor) -> torch.Tensor:
        if los.ndim != 1:
            raise ValueError("GIN LOS path expects rank-1 LOS indices")
        return self.los_embedding_layer(los.long())

    def _is_shared_edge(self, edge_index: torch.Tensor, num_nodes: int) -> bool:
        key = (
            edge_index.device.type,
            edge_index.device.index,
            edge_index.data_ptr(),
            edge_index.size(1),
            num_nodes,
        )
        if self._edge_format_cache is not None and self._edge_format_cache[0] == key:
            return self._edge_format_cache[1]
        is_shared = edge_index.numel() == 0 or int(edge_index.detach().max().cpu().item()) < num_nodes
        self._edge_format_cache = (key, is_shared)
        return is_shared

    def forward(self, x, los, edge_index, **kwargs):
        # initial setting
        if x.ndim == 1:
            batch_size = 1
            x = x.unsqueeze(dim=0)
        elif x.ndim == 2:
            batch_size = x.shape[0]
        else:
            raise ValueError("incorrect x dim")
        
        if self.use_los:
            if los.ndim == 1:
                los_feature = los.unsqueeze(dim=1)
            else:
                raise ValueError(f"Unsupported LOS input rank: {los.ndim}")
        else:
            los_feature = None

        num_nodes = x.shape[1] + (1 if los_feature is not None else 0)

        # entity embedding
        self._validate_x_feature_width(x)
        x_embedded = self.entity_embedding_layer(x.long())
        if los_feature is not None:
            los_embedded = self.encode_los(los.long()).unsqueeze(dim=1)
            x_embedded = torch.cat((x_embedded, los_embedded), dim=1)

        # gin layers
        use_shared_edge = self._is_shared_edge(edge_index, num_nodes)
        node_embeddings = x_embedded if use_shared_edge else x_embedded.reshape(batch_size * num_nodes, -1)
        sum_pooled = []
        for layer in self.gin_layers:
            node_embeddings = layer(node_embeddings, edge_index)
            if use_shared_edge:
                x_sum = torch.sum(node_embeddings, dim=1)
            else:
                x_temp = node_embeddings.reshape(batch_size, num_nodes, -1)
                x_sum = torch.sum(x_temp, dim=1)
            sum_pooled.append(x_sum)
        graph_emb = torch.cat(sum_pooled, dim=1) # [batch, feature_dim * layer_num]

        # classifier
        return self.classifier(graph_emb)
    
