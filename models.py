from dgl.nn.functional import edge_softmax
from dgl import DGLError
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Feature computation h => h'.

    **Linear transformation**:
        :math:`z_i=Wh_i`
    **Edge attention**:
        :math:`e_{ij}=LeakyReLU\ (a^T\cdot(z_i\ ||\ z_j\ ))`
    **Aggregation**:
        :math:`a_{ij}=\exp(e_{ij}\ )/(\sum_{k\in \mathcal{N}(i\ )}\exp(e_{ik}\ ))`

        :math:`h_i'=Relu\ (\sum_{j\in \mathcal{N}(i\ )} {a_{ij}\ z_j\ })`
    """
    def __init__(self, in_dim: int, out_dim: int,
                 feat_drop: float = 0., attn_drop: float = 0.,
                 negative_slope: float = 0.2):
        super(GATLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_s = nn.Linear(out_dim, 1, bias=False)
        self.attn_d = nn.Linear(out_dim, 1, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_s.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_d.weight, gain=gain)

    def forward(self, graph, h):
        with graph.local_scope():
            h = self.feat_drop(h)
            z_s = self.linear(h)
            if graph.is_block:
                z_d = z_s[:graph.number_of_dst_nodes()]
            else:
                z_d = z_s

            # compute edge attention
            es = self.attn_s(z_s)
            ed = self.attn_d(z_d)
            graph.srcdata.update({'z': z_s, 'es': es})
            graph.dstdata.update({'ed': ed})
            graph.apply_edges(fn.u_add_v('es', 'ed', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e('z', 'a', 'm'), fn.sum('m', 'z'))

            return graph.dstdata['z']


class MultiHeadGATLayer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_heads: int,
                 dropout: float = 0.2):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(
                in_dim, out_dim, feat_drop=dropout, attn_drop=0.2))

    def forward(self, graph, h):
        batch = [head(graph, h) for head in self.heads]
        return torch.cat(batch, dim=-1)


class JKNLayer(nn.Module):
    """**Layer-level aggregation**
        :math:`\mathbf{h}_{final}=\phi(\mathbf{h}^1, \mathbf{h}^2, ...\ , \mathbf{h}^L\ )`

    * **Concatenation**: :math:`\phi = ||`

    * **Gated Recurrent Unit**:

        :math:`(\mathbf{f}^l,\mathbf{b}^l) = bidirectional\ GRU\ (\mathbf{h}^l\ )`

        Attention score for layers
        :math:`s^l = a^T\cdot(\mathbf{f}^l\ ||\ \mathbf{b}^l)`

        :math:`\mathbf{h}_{final} = softmax(\mathbf{s})\cdot(\mathbf{f}\ ||\ \mathbf{b})`

    Parameters
    ----------
    in_dim : int
        input feature size.
    out_dim : int
        output feature size.
    n_layers : int
        number of layers in multi-layer skip connection.
    aggregator_type : str
        the aggregation method. It supports "cat" and "gru". The default value is "cat".
    """
    def __init__(self, in_dim: int, out_dim: int,
                 n_layers: int,
                 aggregator_type='cat'):
        super(JKNLayer, self).__init__()
        if aggregator_type not in ['cat', 'gru']:
            raise KeyError('Aggregator type {} not supported.'.format(aggregator_type))
        self.__aggr_type = aggregator_type
        if aggregator_type == 'cat':
            self.linear = nn.Linear(in_dim * n_layers, out_dim, bias=False)
        if aggregator_type == 'gru':
            self.gru = nn.GRU(input_size=in_dim,
                              hidden_size=in_dim,
                              bidirectional=True)
            self.s = nn.Linear(in_dim * 2, 1, bias=False)
            self.linear = nn.Linear(in_dim * 2, out_dim, bias=False)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        if self.__aggr_type == 'gru':
            nn.init.xavier_normal_(self.s.weight, gain=gain)

    def forward(self, h_list):
        if self.__aggr_type == 'cat':
            h_cat = torch.cat(h_list, dim=-1)
            return self.linear(h_cat)
        else:
            # gru
            h_seq, _ = self.gru(torch.stack(h_list, dim=0))
            w = self.s(h_seq)
            w = F.softmax(w, dim=-1)
            h_N = torch.einsum('lni,lnf->nf', w, h_seq)
            return self.linear(h_N)


class Model(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int,
                 n_layers: int = 2, n_heads: int = 2,
                 dropout: float = 0.2, skip_connection='gru'):
        super(Model, self).__init__()
        self.input_layer = MultiHeadGATLayer(in_dim, h_dim, n_heads, dropout)
        if n_layers < 1:
            raise DGLError('Number of layers should be positive integer.')
        hid_dim = h_dim * n_heads
        n_h_layers = n_layers - 1
        self.layers = nn.ModuleList()
        for i in range(n_h_layers):
            self.layers.append(MultiHeadGATLayer(hid_dim, h_dim, n_heads, dropout))
        self.output_layer = JKNLayer(hid_dim, out_dim, n_layers, skip_connection)

    def forward(self, graph, in_feat):
        h = self.input_layer(graph, in_feat)
        h_list = [F.relu(h)]
        for layer in self.layers:
            h = layer(graph, F.relu(h))
            h_list.append(F.relu(h))
        return self.output_layer(h_list)
