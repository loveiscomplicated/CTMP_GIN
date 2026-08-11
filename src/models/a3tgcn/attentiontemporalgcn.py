import torch
from src.models.a3tgcn.temporalgcn import TGCN
from src.models.a3tgcn.temporalgcn import TGCN2

import sys
import os
cur_dir = os.path.dirname(__file__)
parent_dir = os.path.join(cur_dir, '..')
sys.path.append(parent_dir)


class A3TGCN(torch.nn.Module):
    r"""An implementation of the Attention Temporal Graph Convolutional Cell.
    For details see this paper: `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Prediction." <https://arxiv.org/abs/2006.11583>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        num_layers: int,
        device: torch.device = torch.device("cpu"),
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True
    ):
        super(A3TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.device = device
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=self.device))
        torch.nn.init.uniform_(self._attention)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor | None = None,
        H: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_accum = 0
        H_sequence_outputs = [] 
        
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        
        H_previous = H 
        
        for period in range(self.periods):
            X_current = X[:, :, :, period]
            H_current = self._base_tgcn(X_current, edge_index, edge_weight, H_previous)
            H_previous = H_current 
            H_sequence_outputs.append(probs[period] * H_current)

        H_accum = torch.stack(H_sequence_outputs, dim=0).sum(dim=0)

        return H_accum



class A3TGCN2(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Attention Temporal Graph Convolutional Cell.
    For details see this paper: `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Prediction." <https://arxiv.org/abs/2006.11583>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        num_layers (int): Number of GCNConv layers in TGCN module.
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(
        self,
        in_channels: int, 
        out_channels: int,  
        periods: int, 
        batch_size:int,
        device: torch.device = torch.device("cpu"),
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        fuse_gates: bool = False,
        num_layers: int = 1,
        dropout_p: float = 0.0):
        super(A3TGCN2, self).__init__()

        self.in_channels = in_channels  # 2
        self.out_channels = out_channels # 32
        self.periods = periods # 12
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.fuse_gates = bool(fuse_gates)
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.dropout_p = float(dropout_p)
        self.batch_size = batch_size
        self.device = device
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn_layers = torch.nn.ModuleList()
        for layer_idx in range(self.num_layers):
            self._base_tgcn_layers.append(TGCN2(
                in_channels=self.in_channels if layer_idx == 0 else self.out_channels,
                out_channels=self.out_channels,
                batch_size=self.batch_size,
                improved=self.improved,
                cached=self.cached,
                add_self_loops=self.add_self_loops,
                fuse_gates=self.fuse_gates,
            ))
        self._base_tgcn = self._base_tgcn_layers[0]
        self._dropout = torch.nn.Dropout(self.dropout_p)

        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=self.device))
        torch.nn.init.uniform_(self._attention)

    def forward( 
        self, 
        X: torch.FloatTensor,
        edge_index: torch.LongTensor, 
        edge_weight: torch.FloatTensor | None = None,
        H: torch.FloatTensor | None  = None
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_accum = 0
        H_sequence_outputs = [] 
        
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        
        if H is None:
            H_previous = [None for _ in range(self.num_layers)]
        elif isinstance(H, (list, tuple)):
            if len(H) != self.num_layers:
                raise ValueError(f"H must have {self.num_layers} hidden states, got {len(H)}")
            H_previous = list(H)
        elif self.num_layers == 1:
            H_previous = [H]
        else:
            H_previous = [None for _ in range(self.num_layers)]
        
        for period in range(self.periods):
            X_current = X[:, :, :, period] # shape: [batch_size, num_nodes, feature_dim]. 이런 식으로 반복문 돌리면 period와 같은 차원은 없어짐(축소)
            current = X_current
            for layer_idx, layer in enumerate(self._base_tgcn_layers):
                H_current = layer(current, edge_index, edge_weight, H_previous[layer_idx])
                H_previous[layer_idx] = H_current
                current = self._dropout(H_current) if layer_idx < self.num_layers - 1 else H_current
            H_sequence_outputs.append(probs[period] * current)

        H_accum = sum(H_sequence_outputs)

        return H_accum
