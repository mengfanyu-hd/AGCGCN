import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import os
import math


class CHGCNGCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=1):
        super(CHGCNGCN, self).__init__()
        self.nconv = NConv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            if self.order >= 2:
                for k in range(2, self.order + 1):
                    x2 = self.nconv(x1, a)
                    out.append(x2)
                    x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class CHGCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, Mor, global_nodes, regional_nodes, device, n1, n2, n3, n4, support_len,
                 order=1):
        super(CHGCN, self).__init__()
        self.gcn = CHGCNGCN(c_in, c_out, dropout, support_len, order)

        self.regional_nodes = regional_nodes
        self.global_nodes = global_nodes
        self._device = device
        self.Mor = Mor.to(device=self._device)

        '''
        n for \eta in the paper
        '''
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4

    def forward(self, x, support, support_r, Mrg):
        # graph convolution on the original graph
        ho = self.gcn(x, support)
        # graph convolution on the regional graph
        hr = self.gcn(self.Mor.t().float() @ x, support_r)

        # construct the feature matrix of the global graph
        xs = Mrg.t().float() @ self.Mor.t().float() @ x

        # shape (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
        xg = xs.permute(2, 1, 0, 3)
        xg = torch.reshape(xg, (self.global_nodes, -1))
        # print(xg.size())
        xgt = xs.permute(3, 0, 1, 2)
        xgt = torch.reshape(xgt, (-1, self.global_nodes))

        # print(xgt.size())
        Ag_hat = (F.relu(xg @ xgt - 0.5)).detach().cpu().numpy()
        # build adjacency matrix for the global graph
        adj_mxs = [asym_adj(Ag_hat), asym_adj(np.transpose(Ag_hat))]
        supports_g = [torch.tensor(i).to(self._device) for i in adj_mxs]
        supports_g = [F.softmax(i, dim=-1).to(self._device) for i in supports_g]

        # graph convolution on the regional graph
        xg = self.gcn(xs, supports_g)

        # cross-layer information enhancement
        hr = hr + self.n1 * F.relu(Mrg.float() @ xg)
        ho = ho + self.n2 * F.relu(self.Mor.float() @ hr)
        hr = hr + self.n3 * F.relu(self.Mor.t().float() @ ho)
        xg = xg + self.n4 * F.relu(Mrg.t().float() @ hr)

        return ho, hr, xg


"""
STTN
"""


class SSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
                self.head_dim * num_heads == embed_dim
        ), "Embedding dim needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, values, keys, query):
        batch_size, num_nodes, input_window, embed_dim = query.shape

        values = values.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        query = query.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("bqthd,bkthd->bqkth", [queries, keys])

        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=2)

        out = torch.einsum("bqkth,bkthd->bqthd", [attention, values]).reshape(
            batch_size, num_nodes, input_window, self.num_heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=torch.device('cpu')):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features).to(device))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj_mx):
        support = torch.einsum("bnd, dh->bnh", [x, self.weight])
        output = torch.einsum("mn,bnh->bmh", [adj_mx, support])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, device=device)
        self.gc2 = GraphConvolution(nhid, nclass, device=device)
        self.dropout_rate = dropout_rate

    def forward(self, x, adj_mx):
        x = F.relu(self.gc1(x, adj_mx))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.gc2(x, adj_mx)
        return F.log_softmax(x, dim=2)


class STransformer(nn.Module):
    def __init__(self, adj_mx,
                 supports, supports_r, Mrg,
                 c_in, c_out, dropout, Mor, global_nodes, regional_nodes, n1, n2, n3, n4, support_len, order=1,

                 embed_dim=64, num_heads=2,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()

        self.supports = supports
        self.supports_r = supports_r
        self.Mrg = Mrg

        self.dilation_channels = c_in
        self.residual_channels = c_out
        self.dropout = dropout
        self.Mor_mx = Mor
        self.global_nodes = global_nodes
        self.regional_nodes = regional_nodes
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.supports_len = support_len
        self.order = order

        self.conv1 = nn.Conv2d(self.dilation_channels, embed_dim, 1)
        self.conv11 = nn.Conv2d(embed_dim, self.dilation_channels, 1)

        self.device = device
        self.adj_mx = torch.FloatTensor(adj_mx).to(device)
        self.D_S = nn.Parameter(torch.FloatTensor(adj_mx).to(device))
        self.embed_linear = nn.Linear(adj_mx.shape[0], embed_dim)

        self.attention = SSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )

        # self.gcn = GCN(embed_dim, embed_dim * 2, embed_dim, dropout_rate, device=device)

        self.chgcn = CHGCN(self.dilation_channels, self.residual_channels, self.dropout, self.Mor_mx, self.global_nodes,
                           self.regional_nodes, device=torch.device('cuda')
                           , n1=self.n1, n2=self.n2, n3=self.n3, n4=self.n4, support_len=self.supports_len,
                           order=self.order)

        self.norm_adj = nn.InstanceNorm2d(1)

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.fs = nn.Linear(embed_dim, embed_dim)
        self.fg = nn.Linear(embed_dim, embed_dim)

    def forward(self, value, key, query):
        X_CHGCN, xc, xs = self.chgcn(query, self.supports, self.supports_r, self.Mrg)
        X_CHGCN = self.conv1(X_CHGCN)
        X_CHGCN = X_CHGCN.permute(0, 2, 3, 1)

        query = self.conv1(query)
        query = query.permute(0, 2, 3, 1)
        value = query
        key = query

        batch_size, num_nodes, input_windows, embed_dim = query.shape
        D_S = self.embed_linear(self.D_S)
        D_S = D_S.expand(batch_size, input_windows, num_nodes, embed_dim)
        D_S = D_S.permute(0, 2, 1, 3)

        query = query + D_S
        attention = self.attention(value, key, query)

        x = self.dropout_layer(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout_layer(self.norm2(forward + x))

        g = torch.sigmoid(self.fs(U_S) + self.fg(X_CHGCN))
        out = g * U_S + (1 - g) * X_CHGCN

        out = out.permute(0, 3, 1, 2)
        out = self.conv11(out)

        return out, xc, xs


class STTransformerBlock(nn.Module):
    def __init__(self, adj_mx,
                 supports, supports_r, Mrg,
                 c_in, c_out, dropout, Mor, global_nodes, regional_nodes, n1, n2, n3, n4, support_len, order=1,

                 embed_dim=64, num_heads=2, TG_per_day=288,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.STransformer = STransformer(
            adj_mx,
            supports, supports_r, Mrg,
            c_in, c_out, dropout, Mor, global_nodes, regional_nodes, n1, n2, n3, n4, support_len, order=1,

            embed_dim=embed_dim, num_heads=num_heads,
            forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device)

        self.norm1 = nn.LayerNorm(c_in)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, value, key, query):
        x1, xc, xs = self.STransformer(value, key, query)
        x1 = self.norm1((x1 + query).permute(0, 3, 2, 1))
        # x2 = self.dropout_layer(self.norm2(self.TTransformer(x1, x1, x1) + x1))
        x2 = self.dropout_layer(x1.permute(0, 3, 2, 1))
        return x2, xc, xs


class Encoder(nn.Module):
    def __init__(self, adj_mx,
                 supports, supports_r, Mrg,
                 c_in, c_out, dropout, Mor, global_nodes, regional_nodes, n1, n2, n3, n4, support_len, order=1,
                 embed_dim=64, num_layers=3, num_heads=2, TG_per_day=288,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.layers = nn.ModuleList([
            STTransformerBlock(
                adj_mx,
                supports, supports_r, Mrg,
                c_in, c_out, dropout, Mor, global_nodes, regional_nodes, n1, n2, n3, n4, support_len, order=1,

                embed_dim=embed_dim, num_heads=num_heads, TG_per_day=TG_per_day,
                forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device
            )
            for _ in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.dropout_layer(x)
        for layer in self.layers:
            out, xc, xs = layer(out, out, out)
        return out, xc, xs


class Transformer(nn.Module):
    def __init__(self, adj_mx,
                 supports, supports_r, Mrg,
                 c_in, c_out, dropout, Mor, global_nodes, regional_nodes, n1, n2, n3, n4, support_len, order=1,

                 embed_dim=64, num_layers=3, num_heads=2, TG_per_day=288,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu'),

                 ):
        super().__init__()
        self.encoder = Encoder(
            adj_mx,
            supports, supports_r, Mrg,
            c_in, c_out, dropout, Mor, global_nodes, regional_nodes, n1, n2, n3, n4, support_len, order=1,
            embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, TG_per_day=TG_per_day,
            forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device,
        )

    def forward(self, src):
        enc_src, xc, xs = self.encoder(src)
        return enc_src, xc, xs


# torch.autograd.set_detect_anomaly(True)
def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).todense()


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncvl,vw->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class AGCGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        self.adj_mx = data_feature.get('adj_mx')
        self.transformer_adj_mx = self.adj_mx  #transformer
        self.Mor = data_feature.get('Mor_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.regional_nodes = len(self.Mor[0])
        self.global_nodes = config.get('global_nodes', 15)
        print('global_nodes ' + str(self.global_nodes))
        self.feature_dim = data_feature.get('feature_dim', 2)
        super().__init__(config, data_feature)

        self.dropout = config.get('dropout', 0.3)
        self.blocks = config.get('blocks', 4)
        self.layers = config.get('layers', 2)
        self.adjtype = config.get('adjtype', 'doubletransition')
        self.kernel_size = config.get('kernel_size', 2)
        self.nhid = config.get('nhid', 32)
        self.residual_channels = config.get('residual_channels', self.nhid)
        self.dilation_channels = config.get('dilation_channels', self.nhid)
        self.skip_channels = config.get('skip_channels', self.nhid * 8)
        self.end_channels = config.get('end_channels', self.nhid * 16)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.order = config.get('order', 2)
        self.device = config.get('device', torch.device('cpu'))

        # transfer the matrix into tensor format
        self.Mor_mx = torch.from_numpy(self.Mor).detach().to(device=self.device)  # Mapping matrix
        self.ao = torch.from_numpy(self.adj_mx).detach().to(
            device=self.device)  # adjacency matrix for the original graph
        self.adj_mxr = torch.from_numpy(self.Mor.T @ self.adj_mx @ self.Mor).detach().to(
            device=self.device)  # adjacency matrix for the regional graph

        #########################
        self.embed_dim = config.get('embed_dim', 64)
        self.num_layers = config.get('num_layers', 3)
        self.num_heads = config.get('num_heads', 2)
        self.TG_per_day = config.get('TG_in_one_day', 288)  # number of time intevals per day
        self.forward_expansion = config.get('forward_expansion', 4)
        self.dropout_rate = config.get('dropout_rate', 0)
        #########################

        # default values for etas
        self.n1 = config.get('n1', 1)
        self.n2 = config.get('n2', 1)
        self.n3 = config.get('n3', 1)
        self.n4 = config.get('n4', 1)

        # adaptive layer numbers
        self.apt_layer = config.get('apt_layer', True)
        if self.apt_layer:
            self.layers = np.int(
                np.round(np.log((((self.input_window - 1) / (self.blocks * (self.kernel_size - 1))) + 1)) / np.log(2)))
            print('# of layers change to %s' % self.layers)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.bnc = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.gconv = nn.ModuleList()

        # init the mapping matrix M_rg
        # this is a trainable parameter of our proposed model
        self.Mrg = torch.nn.init.kaiming_uniform_(torch.nn.Parameter(
            torch.empty((self.regional_nodes, self.global_nodes), requires_grad=True).to(device=self.device)))

        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        # construct the adjacency matrix according to the seetings
        self.cal_adj(self.adjtype)
        self.supports = [torch.tensor(i).to(self.device) for i in self.adj_mx]  # adjacency matrix of the original graph
        # construct the adjacency matrix of the regional graph
        self.supports_r = [(self.Mor_mx.t().float() @ i.clone().detach() @ self.Mor_mx.float()).to(self.device) for i in
                           self.supports]
        self.supports_r = [F.softmax(i, dim=-1).to(self.device) for i in self.supports_r]

        receptive_field = self.output_dim

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        # self.supports_len = 1
        print('supports_len ' + str(self.supports_len))
        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # print(self.filter_convs[-1])
                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # print(self.gate_convs[-1])
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                self.bnc.append(nn.BatchNorm2d(self.residual_channels))
                self.bns.append(nn.BatchNorm2d(self.residual_channels))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

                # Add the Transformer
                self.transformer = Transformer(
                    self.transformer_adj_mx,
                    self.supports, self.supports_r, self.Mrg,

                    self.dilation_channels, self.residual_channels, self.dropout, self.Mor_mx, self.global_nodes,
                    self.regional_nodes, n1=self.n1, n2=self.n2, n3=self.n3, n4=self.n4, support_len=self.supports_len,
                    order=self.order,

                    embed_dim=self.embed_dim, num_layers=self.num_layers, num_heads=self.num_heads,
                    TG_per_day=self.TG_per_day, forward_expansion=self.forward_expansion,
                    dropout_rate=self.dropout_rate,
                    device=self.device
                )

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.receptive_field = receptive_field
        self._logger.info('receptive_field: ' + str(self.receptive_field))

    def forward(self, batch):
        inputs = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))  # (batch_size, feature_dim, num_nodes, input_window+1)

        # Make elements value range from 0 to 1
        Mrg = F.softmax(F.relu(self.Mrg), dim=1)

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = inputs

        x = self.start_conv(x)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        skip = 0

        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            # (dilation, init_dilation) = self.dilations[i]
            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # (batch_size, residual_channels, num_nodes, self.receptive_field)
            # dilated convolution
            filter = self.filter_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # parametrized skip connection
            s = x
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            s = self.skip_convs[i](s)
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except(Exception):
                skip = 0
            skip = s + skip

            input_transformer = x
            # x, xc, xs = self.gconv[i](x, self.supports, self.supports_r, Mrg)
            x, xc, xs = self.transformer(input_transformer)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            xc = self.bnc[i](xc)
            xs = self.bns[i](xs)
        # process and transfer the output of original graph for prediction

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, xc, xs

    def ortLoss(self, hr):
        '''
        The implementation of L_{ort}
        '''
        # hr = (batch_size, end_channels, num_nodes, self.output_dim)
        (a, b, c, d) = hr.shape
        hr_sum = torch.sum(hr, dim=-1, keepdim=True)  # 在最后一个维度上求和
        hr_mean = hr_sum / d  # 求平均值

        hr = hr_mean.squeeze(3)

        hr = hr.permute(2, 0, 1)
        hr = torch.reshape(hr, (self.global_nodes, -1))
        # print('hr.size')
        # print(hr.size())
        tmpLoss = 0
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(self.global_nodes - 1):
            for j in range(i + 1, self.global_nodes):
                # print(hr[i].size())
                tmpLoss += cos(hr[i], hr[j])

        tmpLoss = tmpLoss / (self.global_nodes * (self.global_nodes - 1) / 2)

        return tmpLoss

    def cal_adj(self, adjtype):
        if adjtype == "scalap":
            self.adj_mx = [calculate_scaled_laplacian(self.adj_mx)]
        elif adjtype == "normlap":
            self.adj_mx = [calculate_normalized_laplacian(self.adj_mx).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            self.adj_mx = [sym_adj(self.adj_mx)]
        elif adjtype == "transition":
            self.adj_mx = [asym_adj(self.adj_mx)]
        elif adjtype == "doubletransition":
            self.adj_mx = [asym_adj(self.adj_mx), asym_adj(np.transpose(self.adj_mx))]
        elif adjtype == "identity":
            self.adj_mx = [np.diag(np.ones(self.adj_mx.shape[0])).astype(np.float32)]
        else:
            assert 0, "adj type not defined"

    def calculate_loss(self, batch):
        y_true = batch['y']

        y_predicted, xr, xg = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        Mrg = F.softmax(F.relu(self.Mrg), dim=1)  # -0.5

        no = self.Mor_mx.float() @ xr
        ho = no.permute(2, 1, 0, 3)
        ho = torch.reshape(ho, (self.num_nodes, -1))
        # print(hr.size())
        hot = no.permute(3, 0, 1, 2)
        hot = torch.reshape(hot, (-1, self.num_nodes))
        # reconstruct the adjacency matrix of the original graph
        ao_hat = torch.sigmoid(ho.float() @ hot.float())

        nr = Mrg.float() @ xg
        hr = nr.permute(2, 1, 0, 3)
        hr = torch.reshape(hr, (self.regional_nodes, -1))
        hrt = nr.permute(3, 0, 1, 2)
        hrt = torch.reshape(hrt, (-1, self.regional_nodes))
        # reconstruct the adjacency matrix of the original graph
        ar_hat = torch.sigmoid(hr.float() @ hrt.float())

        preLoss = loss.masked_mae_torch(y_predicted, y_true, 0)
        recLoss0 = F.binary_cross_entropy(ao_hat, self.ao, reduction='mean')
        recLoss = F.binary_cross_entropy(ar_hat, self.adj_mxr.float(), reduction='mean')
        ortLoss = self.ortLoss(xg)

        # self._logger.info('fine_loss: {0} bce_loss:{1} bce_loss0:{2} ortLoss:{3}'.format(preLoss,recLoss,recLoss0,ortLoss))
        return preLoss + 0.01 * recLoss + 0.01 * recLoss0 + 0.1 * ortLoss

    def predict(self, batch):
        return self.forward(batch)
