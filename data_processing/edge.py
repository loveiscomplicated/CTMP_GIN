import torch
import pickle
from copy import deepcopy
from torch_geometric.utils import to_undirected

def fully_connected_edge_index(num_nodes, self_loops=False):
    '''
    이름 그대로, num_nodes 즉 변수 개수만 알면 됨
    '''
    nodes = torch.arange(num_nodes)
    row, col = torch.meshgrid(nodes, nodes, indexing="ij")
    edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)
    if not self_loops:
        mask = row != col
        edge_index = edge_index[:, mask.reshape(-1)]
    return edge_index

def fully_connected_edge_index_batched(num_nodes, batch_size, self_loops=False):
    '''
    batched edge_index는 결국 옆으로 이어붙인 것일 뿐, 즉 shape: [2, sum(num_edges{i})]
    '''
    single = fully_connected_edge_index(num_nodes=num_nodes)
    batch_list = [single for i in range(batch_size)]
    return torch.concatenate(batch_list, dim=1)


def mi_edge_index_single(mi_dict, top_k=6, threshold=0.01, pruning_ratio=0.5, return_edge_attr=False):
    """
    개선된 MI 기반 그래프 생성 함수 (Strategies 1, 2, 3 적용)
    
    Args:
        mi_dict_path (str): 피클 파일 경로
        top_k (int): 상위 k개 선택
        threshold (float): [Strategy 2] MI 값이 이보다 작으면 연결하지 않음 (기본값 0.01)
        pruning_ratio (float): [Strategy 3] In-Degree가 전체 노드 수의 이 비율을 넘으면 하위 엣지 삭제 (기본값 0.5 = 50%)
        return_edge_attr (bool): 가중치 반환 여부
    """
    cols = list(mi_dict.keys())
    num_nodes = len(cols)
    col_to_idx = {c: i for i, c in enumerate(cols)}

    # 임시 저장을 위한 리스트 (source, target, weight)
    raw_edges = []

    # 1. 초기 유향 엣지 생성 (Threshold & Top-k 적용)
    for src in cols:
        series = mi_dict[src]

        # 유효 변수 필터링 & 자기 자신 제외
        series = series[series.index.isin(cols)]
        if src in series.index:
            series = series.drop(index=src)

        # [Strategy 2] Threshold 적용 (너무 약한 관계 끊기)
        series = series[series >= threshold]

        # Top-k 선택
        top_neighbors = series.head(top_k)

        src_idx = col_to_idx[src]
        for dst, w in top_neighbors.items():
            dst_idx = col_to_idx[dst]
            raw_edges.append((src_idx, dst_idx, float(w)))

    # 2. [Strategy 3] 구조적 Pruning (Hub 노드 견제)
    # Target 노드별로 엣지를 모아서 In-Degree가 너무 높으면 약한 것부터 잘라냅니다.
    
    # Target별로 그룹화: {dst_idx: [(src, dst, w), ...]}
    edges_by_target = {}
    for edge in raw_edges:
        dst = edge[1]
        if dst not in edges_by_target:
            edges_by_target[dst] = []
        edges_by_target[dst].append(edge)
    
    final_edges = []
    max_in_degree = int(num_nodes * pruning_ratio) # 허용 가능한 최대 In-Degree (예: 60개 중 30개)

    for dst, edges in edges_by_target.items():
        # 만약 특정 노드(예: STFIPS)로 들어오는 엣지가 너무 많다면?
        if len(edges) > max_in_degree:
            # 가중치(MI) 기준 내림차순 정렬 후 상위 N개만 남김
            edges.sort(key=lambda x: x[2], reverse=True)
            kept_edges = edges[:max_in_degree]
            final_edges.extend(kept_edges)
        else:
            final_edges.extend(edges)

    # 텐서 변환 준비
    if not final_edges:
        print("⚠️ 주의: 조건에 맞는 엣지가 하나도 없습니다. Threshold를 낮추세요.")
        return torch.empty((2, 0), dtype=torch.long)

    src_list, dst_list, weight_list = zip(*final_edges)
    
    # Directed Edge Index 생성
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(weight_list, dtype=torch.float) if return_edge_attr else None

    # 3. [Strategy 1] 무방향(Undirected) 그래프로 변환
    # A->B가 있으면 B->A도 생성 (정보 흐름 개선)
    # to_undirected는 중복된 엣지는 제거하고, 양방향을 보장해줍니다.
    if return_edge_attr:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=num_nodes)
        return edge_index, edge_attr
    else:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        return edge_index

def mi_edge_index_batched(batch_size, num_nodes, mi_ad_dict, mi_dis_dict, top_k=6, threshold=0.01, pruning_ratio=0.5, return_edge_attr=False, edge_attr_single=None):
    """
    배치 처리를 위한 Wrapper 함수
    """
    if return_edge_attr:
        single_ad, edge_attr_ad= mi_edge_index_single(
            mi_dict=mi_ad_dict, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )
    else:
        single_ad = mi_edge_index_single(
            mi_dict=mi_ad_dict, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )
    
    if return_edge_attr:
        single_dis, edge_attr_dis= mi_edge_index_single(
            mi_dict=mi_dis_dict, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )
    else:
        single_dis = mi_edge_index_single(
            mi_dict=mi_dis_dict, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )

    edge_list = []
    attr_list = []

    for g in range(batch_size):
        offset = num_nodes * g
        edge_i = single_ad + offset
        edge_list.append(edge_i)

        if return_edge_attr:
            attr_list.append(edge_attr_ad)
    
    offset_ad = offset
    
    for g in range(batch_size):
        offset = num_nodes * g + offset_ad
        edge_i = single_dis + offset
        edge_list.append(edge_i)

        if return_edge_attr:
            attr_list.append(edge_attr_dis)

    batched_edge_index = torch.cat(edge_list, dim=1)
    if return_edge_attr:
        batched_attr_list = torch.cat(attr_list, dim=0)
        return batched_edge_index, batched_attr_list
    return batched_edge_index
    

def mi_edge_index_batched_for_baseline(batch_size, num_nodes, mi_avg_dict, top_k=6, threshold=0.01, pruning_ratio=0.5, return_edge_attr=False, edge_attr_single=None):
    """
    static batched edge index
    """
    batch_size_d = batch_size

    if return_edge_attr:
        single, edge_attr= mi_edge_index_single(
            mi_dict=mi_avg_dict, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )
    else:
        single = mi_edge_index_single(
            mi_dict=mi_avg_dict, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )

    edge_list = []
    attr_list = []

    for g in range(batch_size_d):
        offset = num_nodes * g
        edge_i = single + offset
        edge_list.append(edge_i)

        if return_edge_attr:
            attr_list.append(edge_attr)
    
    batched_edge_index = torch.cat(edge_list, dim=1)

    if return_edge_attr:
        batched_attr_list = torch.cat(attr_list, dim=0)
        return batched_edge_index, batched_attr_list
    
    return batched_edge_index