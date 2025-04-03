from typing import Union, Optional, List, Dict
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, PairTensor

import torch
from torch import Tensor
from torch_geometric.nn import GINEConv as BaseGINEConv, GINConv as BaseGINConv, LEConv as BaseLEConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree
from torch_scatter import scatter


class GINConv(BaseGINConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                edge_atten: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor): 
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r 

        return self.nn(out)

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if edge_atten is not None:
            return x_j * edge_atten
        else:
            return x_j


class GINEConv(BaseGINEConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                edge_atten: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        m = (x_j + edge_attr).relu()

        if edge_atten is not None:
            return m * edge_atten
        else:
            return m


class LEConv(BaseLEConv):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, edge_atten: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x = (x, x)

        a = self.lin1(x[0])
        b = self.lin2(x[1])

        out = self.propagate(edge_index, a=a, b=b, edge_weight=edge_weight, edge_atten=edge_atten, size=None)

        return out + self.lin3(x[1])

    def message(self, a_j: Tensor, b_i: Tensor, edge_weight: OptTensor, edge_atten: OptTensor = None) -> Tensor:
        out = a_j - b_i
        m = out if edge_weight is None else out * edge_weight.view(-1, 1)

        if edge_atten is not None:
            return m * edge_atten
        else:
            return m


def aggregate_sum(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='sum')


def aggregate_mean(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='mean')


def aggregate_min(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='min')


def aggregate_max(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='max')


def aggregate_var(src, index, dim_size):
    mean = aggregate_mean(src, index, dim_size)
    mean_squares = aggregate_mean(src * src, index, dim_size)
    return mean_squares - mean * mean


def aggregate_std(src, index, dim_size):
    return torch.sqrt(torch.relu(aggregate_var(src, index, dim_size)) + 1e-5)


AGGREGATORS = {
    'sum': aggregate_sum,
    'mean': aggregate_mean,
    'min': aggregate_min,
    'max': aggregate_max,
    'var': aggregate_var,
    'std': aggregate_std,
}


def scale_identity(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src


def scale_amplification(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (torch.log(deg + 1) / avg_deg['log'])


def scale_attenuation(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg['log'] / torch.log(deg + 1)
    scale[deg == 0] = 1
    return src * scale


def scale_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (deg / avg_deg['lin'])


def scale_inverse_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg['lin'] / deg
    scale[deg == 0] = 1
    return src * scale


SCALERS = {
    'identity': scale_identity,
    'amplification': scale_amplification,
    'attenuation': scale_attenuation,
    'linear': scale_linear,
    'inverse_linear': scale_inverse_linear
}
