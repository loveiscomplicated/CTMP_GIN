# CTMP-GIN

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv

cur_dir = os.path.dirname(__file__)
par_dir = os.path.join(cur_dir, '..')
sys.path.append(par_dir)

from src.models.entity_embedding import EntityEmbeddingBatch3


'''class GatedFusion(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, A, B, C):
        # A,B,C: [B, out_dim]
        x = torch.cat([A, B, C], dim=-1)        # [B, 3*out_dim]
        w = F.softmax(self.score(x), dim=-1)    # [B, 3]
        A = self.dropout(A)
        B = self.dropout(B)
        C = self.dropout(C)
        fused = w[:, 0:1]*A + w[:, 1:2]*B + w[:, 2:3]*C
        return fused, w
'''

def seperate_x(x: torch.Tensor, ad_idx_t, dis_idx_t, device):
    ad_tensor = torch.index_select(x, dim=1, index=ad_idx_t) # [B, 60, F]
    dis_tensor = torch.index_select(x, dim=1, index=dis_idx_t) # [B, 60, F]
    return torch.concatenate((ad_tensor, dis_tensor), dim=0) # [B*2, 60, F]

class GatedFusion(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, A, B, C):
        x = torch.cat([A, B, C], dim=-1)
        logits = self.score(x)                 # [B,3]
        w = F.softmax(logits, dim=-1)
        A = self.dropout(A); B = self.dropout(B); C = self.dropout(C)
        fused = w[:, 0:1]*A + w[:, 1:2]*B + w[:, 2:3]*C
        return fused, w, logits
    

class CTMPGIN(nn.Module):
    def __init__(self, 
                 col_info,
                 embedding_dim, 
                 gin_hidden_channel, 
                 gin_1_layers, 
                 gin_hidden_channel_2, # 오류를 피하기 위해서는 일단 첫번째 gin과 동일한 차원이어야 함, 만약 다르게 하려면 projection 추가해야 함
                 gin_2_layers, 
                 device,
                 dropout_p = 0.2,
                 los_embedding_dim=8, 
                 max_los=37,
                 train_eps=True, 
                 gate_hidden_ch=None):
        super().__init__()
        self.device = device
        self.dropout_p = dropout_p

        self.col_list, self.col_dims, self.ad_col_index, self.dis_col_index = col_info
        self.ad_idx_t = torch.tensor(self.ad_col_index, dtype=torch.long, device=self.device)
        self.dis_idx_t = torch.tensor(self.dis_col_index, dtype=torch.long, device=self.device)

        self.gin_hidden_channel = gin_hidden_channel
        self.gin_hidden_channel_2 = gin_hidden_channel_2

        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, embedding_dim=embedding_dim)
        self.embed_los = EntityEmbeddingBatch3(col_dims=[max_los + 1], embedding_dim=los_embedding_dim)

        # GIN_1
        gin_nn_input = nn.Sequential(
             nn.Linear(embedding_dim, gin_hidden_channel),
             nn.LayerNorm(gin_hidden_channel),
             nn.ReLU(),

             nn.Linear(gin_hidden_channel, gin_hidden_channel) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        gin_nn = nn.Sequential(
             nn.Linear(gin_hidden_channel, gin_hidden_channel),
             nn.LayerNorm(gin_hidden_channel),
             nn.ReLU(),

             nn.Linear(gin_hidden_channel, gin_hidden_channel) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        # GIN_2
        gin_nn_input2 = nn.Sequential(
             nn.Linear(gin_hidden_channel, gin_hidden_channel_2), # ad, dis 동시에 들어가므로, input 차원이 두 배
             nn.LayerNorm(gin_hidden_channel_2),
             nn.ReLU(),

             nn.Linear(gin_hidden_channel_2, gin_hidden_channel_2) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        gin_nn2 = nn.Sequential(
             nn.Linear(gin_hidden_channel_2, gin_hidden_channel_2),
             nn.LayerNorm(gin_hidden_channel_2),
             nn.ReLU(),

             nn.Linear(gin_hidden_channel_2, gin_hidden_channel_2) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )
        
        # GIN_1
        self.gin_1 = nn.ModuleList()
        gin1_input = GINConv(gin_nn_input, train_eps=train_eps)
        self.gin_1.append(gin1_input)
        for _ in range(gin_1_layers - 1):
            gin_hidden_layer = GINConv(gin_nn, train_eps=train_eps)
            self.gin_1.append(gin_hidden_layer)

        # GIN_2        
        self.gin_2 = nn.ModuleList()
        gin2_input = GINEConv(gin_nn_input2, train_eps=train_eps, edge_dim=los_embedding_dim)
        self.gin_2.append(gin2_input)
        for _ in range(gin_2_layers - 1):
            gin_hidden_layer = GINEConv(gin_nn2, train_eps=train_eps, edge_dim=los_embedding_dim)
            self.gin_2.append(gin_hidden_layer)

        d = gin_hidden_channel
        self.fuse_dim = d

        self.proj_ad = nn.Linear(gin_hidden_channel, d)
        self.proj_dis = nn.Linear(gin_hidden_channel, d)
        self.proj_merged = nn.Linear(gin_hidden_channel_2, d)

        self.gated_fusion = GatedFusion(
            in_dim=3*self.fuse_dim,
            out_dim=self.fuse_dim,
            hidden_dim=gate_hidden_ch,
            dropout=dropout_p
        )

        self.classifier_b = nn.Sequential(
            nn.Linear(self.fuse_dim, self.fuse_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.fuse_dim * 2, 1)
            )

    def forward(self, x, los, edge_index, **kwargs):
        batch_size = x.shape[0]
        num_nodes = len(self.ad_idx_t)

        # x_batch shape: [batch_size, num_var(=72)]
        x_embedded = self.entity_embedding_layer(x) # shape: [batch, num_var, feature_dim]

        # process: [batch * 2, num_nodes, feature_dim]으로 변환하기
        # 위와 같이 되어야 함: 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, ...
        x_seperated = seperate_x(x=x_embedded, # [B*2, 60, 32]
                                 ad_idx_t=self.ad_idx_t, 
                                 dis_idx_t=self.dis_idx_t, 
                                 device=self.device)

        # GIN_1에 입력
        x_flatten = x_seperated.reshape(batch_size * 2 * num_nodes, -1)

        x_after_gin = x_flatten
        for layer in self.gin_1:
            x_after_gin = layer(x_after_gin, edge_index) # [B * 2 * N, F(32)]
            x_graph = x_after_gin.reshape(batch_size * 2, num_nodes, self.gin_hidden_channel) # [B * 2, N, F]
            x_sum = torch.mean(x_graph, dim=1) # [B * 2, N(60), F(32)] --> [B * 2, F(32)]
        
        ad_dis_emb= x_sum.clone() # [B * 2, F(32)]
        ad_emb = ad_dis_emb[:batch_size, :]
        dis_emb = ad_dis_emb[batch_size:, :]
        
        edge_index_2, edge_attr = self.get_new_edge(edge_index=edge_index, los=los, batch_size=batch_size)
        # GIN_2에 입력
        for layer in self.gin_2:
            x_after_gin = layer(x=x_after_gin, edge_index=edge_index_2, edge_attr=edge_attr)
            x_graph = x_after_gin.reshape(batch_size * 2, num_nodes, self.gin_hidden_channel_2)
            x_sum = torch.mean(x_graph, dim=1)

        merged_all = x_sum  # [2B, F2]
        merged = 0.5 * (merged_all[:batch_size] + merged_all[batch_size:])  # [B, F2]
        merged_f = self.proj_merged(merged)  # [B, d]

        # merged 만든 뒤
        ad_f = self.proj_ad(ad_emb)          # [B, d]
        dis_f = self.proj_dis(dis_emb)       # [B, d]

        fused, w, logits = self.gated_fusion(ad_f, dis_f, merged_f)
        logit = self.classifier_b(fused)

        return logit

    def get_new_edge(self, edge_index, los, batch_size):
        device = edge_index.device
        num_nodes = len(self.ad_col_index)
        new_edge_index = self.get_edge_index_2(edge_index=edge_index, num_nodes=num_nodes, batch_size=batch_size)
        new_edge_attr = self.get_edge_attr(los=los, edge_index=edge_index, batch_size=batch_size, num_nodes=num_nodes)
        new_edge_index = new_edge_index.to(device)
        new_edge_attr = new_edge_attr.to(device)
        return new_edge_index, new_edge_attr
        
    def get_edge_index_2(self, edge_index, num_nodes, batch_size):
        '''
        Args:
            edge_index: ad, dis 이어져 있는 edge_index, gin_1에서 사용했던 것
        '''
        merged_num_nodes = num_nodes * batch_size
        start_node = torch.arange(0, merged_num_nodes) # [batch_size * num_nodes] == [merged_num_nodes]
        end_node = start_node + merged_num_nodes       # [batch_size * num_nodes] == [merged_num_nodes]
        
        start_node = start_node.unsqueeze(dim=0)       # [1, merged_num_nodes]
        end_node = end_node.unsqueeze(dim=0)           # [1, merged_num_nodes]
        new_edge_index = torch.cat((start_node, end_node), dim = 0) # [2, merged_num_nodes]
        new_edge_index = new_edge_index.to(edge_index.device)
        return torch.cat((edge_index, new_edge_index), dim=1) # # [2, 원래 edge_index + merged_num_nodes]
    
    def get_edge_attr(self, los, edge_index, batch_size, num_nodes):
        """
        edge_index: internal edges (E_internal)
        los: (B,) with values in [1..max_los]
        returns: (E_internal + B*num_nodes, los_embedding_dim)
        """
        device = edge_index.device
        E_internal = edge_index.size(1)
        E_cross = batch_size * num_nodes  # new_edge_index.size(1)와 동일해야 함

        # 1) internal edges -> NONE token (0)
        none_idx = torch.zeros(E_internal, dtype=torch.long, device=device)   # (E_internal,)
        edge_attr_internal = self.embed_los(none_idx)                         # (E_internal, D)

        # 2) cross edges -> LOS token (1..max_los), sample별로 num_nodes번 반복
        los = los.view(batch_size).to(device).long()                          # (B,)
        los_idx = los.repeat_interleave(num_nodes)                            # (B*N,) = (E_cross,)
        edge_attr_cross = self.embed_los(los_idx)                             # (E_cross, D)

        return torch.cat([edge_attr_internal, edge_attr_cross], dim=0)        # (E_total, D)

class CtmpGIN_return_emb(nn.Module):
    def __init__(self, 
                 col_info,
                 embedding_dim, 
                 gin_hidden_channel, 
                 gin_1_layers, 
                 gin_hidden_channel_2, # 오류를 피하기 위해서는 일단 첫번째 gin과 동일한 차원이어야 함, 만약 다르게 하려면 projection 추가해야 함
                 gin_2_layers, 
                 device,
                 dropout_p = 0.2,
                 los_embedding_dim=8, 
                 max_los=37,
                 train_eps=True, 
                 gate_hidden_ch=None):
        super().__init__()
        self.device = device
        self.dropout_p = dropout_p

        self.col_list, self.col_dims, self.ad_col_index, self.dis_col_index = col_info
        self.ad_idx_t = torch.tensor(self.ad_col_index, dtype=torch.long, device=self.device)
        self.dis_idx_t = torch.tensor(self.dis_col_index, dtype=torch.long, device=self.device)

        self.gin_hidden_channel = gin_hidden_channel
        self.gin_hidden_channel_2 = gin_hidden_channel_2

        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, embedding_dim=embedding_dim)
        self.embed_los = EntityEmbeddingBatch3(col_dims=[max_los + 1], embedding_dim=los_embedding_dim)

        # GIN_1
        gin_nn_input = nn.Sequential(
             nn.Linear(embedding_dim, gin_hidden_channel),
             nn.LayerNorm(gin_hidden_channel),
             nn.ReLU(),

             nn.Linear(gin_hidden_channel, gin_hidden_channel) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        gin_nn = nn.Sequential(
             nn.Linear(gin_hidden_channel, gin_hidden_channel),
             nn.LayerNorm(gin_hidden_channel),
             nn.ReLU(),

             nn.Linear(gin_hidden_channel, gin_hidden_channel) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        # GIN_2
        gin_nn_input2 = nn.Sequential(
             nn.Linear(gin_hidden_channel, gin_hidden_channel_2), # ad, dis 동시에 들어가므로, input 차원이 두 배
             nn.LayerNorm(gin_hidden_channel_2),
             nn.ReLU(),

             nn.Linear(gin_hidden_channel_2, gin_hidden_channel_2) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        gin_nn2 = nn.Sequential(
             nn.Linear(gin_hidden_channel_2, gin_hidden_channel_2),
             nn.LayerNorm(gin_hidden_channel_2),
             nn.ReLU(),

             nn.Linear(gin_hidden_channel_2, gin_hidden_channel_2) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )
        
        # GIN_1
        self.gin_1 = nn.ModuleList()
        gin1_input = GINConv(gin_nn_input, train_eps=train_eps)
        self.gin_1.append(gin1_input)
        for _ in range(gin_1_layers - 1):
            gin_hidden_layer = GINConv(gin_nn, train_eps=train_eps)
            self.gin_1.append(gin_hidden_layer)

        # GIN_2        
        self.gin_2 = nn.ModuleList()
        gin2_input = GINEConv(gin_nn_input2, train_eps=train_eps, edge_dim=los_embedding_dim)
        self.gin_2.append(gin2_input)
        for _ in range(gin_2_layers - 1):
            gin_hidden_layer = GINEConv(gin_nn2, train_eps=train_eps, edge_dim=los_embedding_dim)
            self.gin_2.append(gin_hidden_layer)

        d = gin_hidden_channel
        self.fuse_dim = d

        self.proj_ad = nn.Linear(gin_hidden_channel, d)
        self.proj_dis = nn.Linear(gin_hidden_channel, d)
        self.proj_merged = nn.Linear(gin_hidden_channel_2, d)

        self.gated_fusion = GatedFusion(
            in_dim=3*self.fuse_dim,
            out_dim=self.fuse_dim,
            hidden_dim=gate_hidden_ch,
            dropout=dropout_p
        )

        self.classifier_b = nn.Sequential(
            nn.Linear(self.fuse_dim, self.fuse_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.fuse_dim * 2, 1)
            )

    def forward(self, x, los, edge_index):
        batch_size = x.shape[0]
        num_nodes = len(self.ad_idx_t)

        # x_batch shape: [batch_size, num_var(=72)]
        x_embedded = self.entity_embedding_layer(x) # shape: [batch, num_var, feature_dim]

        # process: [batch * 2, num_nodes, feature_dim]으로 변환하기
        # 위와 같이 되어야 함: 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, ...
        x_seperated = seperate_x(x=x_embedded, # [B*2, 60, 32]
                                 ad_idx_t=self.ad_idx_t, 
                                 dis_idx_t=self.dis_idx_t, 
                                 device=self.device)

        # GIN_1에 입력
        x_flatten = x_seperated.reshape(batch_size * 2 * num_nodes, -1)

        x_after_gin = x_flatten
        for layer in self.gin_1:
            x_after_gin = layer(x_after_gin, edge_index) # [B * 2 * N, F(32)]
            x_graph = x_after_gin.reshape(batch_size * 2, num_nodes, self.gin_hidden_channel) # [B * 2, N, F]
            x_sum = torch.mean(x_graph, dim=1) # [B * 2, N(60), F(32)] --> [B * 2, F(32)]
        
        ad_dis_emb= x_sum.clone() # [B * 2, F(32)]
        ad_emb = ad_dis_emb[:batch_size, :]
        dis_emb = ad_dis_emb[batch_size:, :]
        
        edge_index_2, edge_attr = self.get_new_edge(edge_index=edge_index, los=los, batch_size=batch_size)
        # GIN_2에 입력
        for layer in self.gin_2:
            x_after_gin = layer(x=x_after_gin, edge_index=edge_index_2, edge_attr=edge_attr)
            x_graph = x_after_gin.reshape(batch_size * 2, num_nodes, self.gin_hidden_channel_2)
            x_sum = torch.mean(x_graph, dim=1)

        merged_all = x_sum  # [2B, F2]
        merged = 0.5 * (merged_all[:batch_size] + merged_all[batch_size:])  # [B, F2]
        merged_f = self.proj_merged(merged)  # [B, d]

        # merged 만든 뒤
        ad_f = self.proj_ad(ad_emb)          # [B, d]
        dis_f = self.proj_dis(dis_emb)       # [B, d]

        # 변경:
        fused, w, logits = self.gated_fusion(ad_f, dis_f, merged_f)
        return fused, w, ad_f, dis_f, merged_f, logits


    def get_new_edge(self, edge_index, los, batch_size):
        device = edge_index.device
        num_nodes = len(self.ad_col_index)
        new_edge_index = self.get_edge_index_2(edge_index=edge_index, num_nodes=num_nodes, batch_size=batch_size)
        new_edge_attr = self.get_edge_attr(los=los, edge_index=edge_index, batch_size=batch_size, num_nodes=num_nodes)
        new_edge_index = new_edge_index.to(device)
        new_edge_attr = new_edge_attr.to(device)
        return new_edge_index, new_edge_attr
        
    def get_edge_index_2(self, edge_index, num_nodes, batch_size):
        '''
        Args:
            edge_index: ad, dis 이어져 있는 edge_index, gin_1에서 사용했던 것
        '''
        merged_num_nodes = num_nodes * batch_size
        start_node = torch.arange(0, merged_num_nodes) # [batch_size * num_nodes] == [merged_num_nodes]
        end_node = start_node + merged_num_nodes       # [batch_size * num_nodes] == [merged_num_nodes]
        
        start_node = start_node.unsqueeze(dim=0)       # [1, merged_num_nodes]
        end_node = end_node.unsqueeze(dim=0)           # [1, merged_num_nodes]
        new_edge_index = torch.cat((start_node, end_node), dim = 0) # [2, merged_num_nodes]
        new_edge_index = new_edge_index.to(edge_index.device)
        return torch.cat((edge_index, new_edge_index), dim=1) # # [2, 원래 edge_index + merged_num_nodes]
    
    def get_edge_attr(self, los, edge_index, batch_size, num_nodes):
        """
        edge_index: internal edges (E_internal)
        los: (B,) with values in [1..max_los]
        returns: (E_internal + B*num_nodes, los_embedding_dim)
        """
        device = edge_index.device
        E_internal = edge_index.size(1)
        E_cross = batch_size * num_nodes  # new_edge_index.size(1)와 동일해야 함

        # 1) internal edges -> NONE token (0)
        none_idx = torch.zeros(E_internal, dtype=torch.long, device=device)   # (E_internal,)
        edge_attr_internal = self.embed_los(none_idx)                         # (E_internal, D)

        # 2) cross edges -> LOS token (1..max_los), sample별로 num_nodes번 반복
        los = los.view(batch_size).to(device).long()                          # (B,)
        los_idx = los.repeat_interleave(num_nodes)                            # (B*N,) = (E_cross,)
        edge_attr_cross = self.embed_los(los_idx)                             # (E_cross, D)

        return torch.cat([edge_attr_internal, edge_attr_cross], dim=0)        # (E_total, D)



        

        