import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GINConv

from src.models.entity_embedding import EntityEmbeddingBatch3

class GIN(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 col_info, 
                 gin_dim, 
                 gin_layer_num, 
                 num_classes, 
                 train_eps=True,
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.col_dims = list(col_info[1]) # col_info: (col_list, col_dims, ad_col_index, dis_col_index) 
        self.col_dims.append(37) # LOS needs to be included in GIN, as it's excluded in col_info.
        self.gin_dim = gin_dim
        self.gin_layer_num = gin_layer_num
        self.train_eps = train_eps
        self.num_classes = num_classes

        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, 
                                                            embedding_dim=embedding_dim)
        
        gin_nn_input = nn.Sequential(
             nn.Linear(embedding_dim, gin_dim),
             nn.LayerNorm(gin_dim),
             nn.ReLU(),

             nn.Linear(gin_dim, gin_dim) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        gin_nn = nn.Sequential(
             nn.Linear(gin_dim, gin_dim),
             nn.LayerNorm(gin_dim),
             nn.ReLU(),

             nn.Linear(gin_dim, gin_dim) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        self.gin_layers = nn.ModuleList()

        gin_layer1 = GINConv(nn=gin_nn_input, eps=0, train_eps=self.train_eps)
        self.gin_layers.append(gin_layer1)
        
        for _ in range(self.gin_layer_num - 1):
            gin_layer_hidden = GINConv(nn=gin_nn, eps=0, train_eps=self.train_eps)
            self.gin_layers.append(gin_layer_hidden)

        # 분류기 레이어 정의
        out_dim = 1 if self.num_classes == 2 else self.num_classes
        self.classifier_dim = self.gin_dim * self.gin_layer_num
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_dim, self.classifier_dim * 2),
            nn.ReLU(),
            nn.Linear(self.classifier_dim * 2, out_dim)
        )

    def forward(self, x, los, edge_index, **kwargs):
        # initial setting
        if x.ndim == 1:
            batch_size = 1
            x = x.unsqueeze(dim=0)
        elif x.ndim == 2:
            batch_size = x.shape[0]
        else:
            raise ValueError("incorrect x dim")
        
        los = los.unsqueeze(dim=1)
        x = torch.cat((x, los), dim=1)

        num_nodes = x.shape[1]

        # entity embedding
        x_embedded = self.entity_embedding_layer(x) # [batch, num_var, entity_emb_dim]

        # gin layers
        node_embeddings = x_embedded.reshape(batch_size * num_nodes, -1) # [batch * num_var, entity_emb_dim]
        sum_pooled = []
        for layer in self.gin_layers:
            node_embeddings = layer(node_embeddings, edge_index) # [batch * num_var, feature_dim]
            x_temp = node_embeddings.reshape(batch_size, num_nodes, -1) # [batch, num_var, feature_dim]
            x_sum = torch.sum(x_temp, dim=1) # [batch, feature_dim]
            sum_pooled.append(x_sum)
        graph_emb = torch.cat(sum_pooled, dim=1) # [batch, feature_dim * layer_num]

        # classifier
        return self.classifier(graph_emb)
    