"""Uni-Sign pose encoder (4-stream ST-GCN) for the SP-10 RTMPose rerun.
Graph / ST-GCN code below is copied from github.com/ZechengLi19/Uni-Sign (stgcn_layers/gcn_utils.py, stgcn_block.py) so that
state-dict key names match the released checkpoints; the checkpoints are CC-BY-NC-4.0 (non-commercial research use).
UniPoseEncoder reproduces the pose branch of models.Uni_Sign.forward (no RGB branch, no mT5)."""
import math, copy
import numpy as np, torch, torch.nn as nn



class Graph:
    """The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self, layout='custom', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # 'body', 'left', 'right', 'mouth', 'face'
        # if layout == 'custom_hand21':
        if layout == 'left' or layout == 'right':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [0, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [0, 9],
                [9, 10],
                [10, 11],
                [11, 12],
                [0, 13],
                [13, 14],
                [14, 15],
                [15, 16],
                [0, 17],
                [17, 18],
                [18, 19],
                [19, 20],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'body':
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [3, 5],
                [5, 7],
                [4, 6],
                [6, 8],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'face_all':
            self.num_node = 9 + 8 + 1
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[i, i + 1] for i in range(9 - 1)] + \
                             [[i, i + 1] for i in range(9, 9 + 8 - 1)] + \
                             [[9 + 8 - 1, 9]] + \
                             [[17, i] for i in range(17)]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = self.num_node - 1

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (
                                self.hop_dis[j, self.center]
                                == self.hop_dis[i, self.center]
                            ):
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif (
                                self.hop_dis[j, self.center]
                                > self.hop_dis[i, self.center]
                            ):
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

class GCN_unit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        A,
        adaptive=True,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        assert A.size(0) == self.kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )
        self.adaptive = adaptive
        # print(self.adaptive)
        if self.adaptive:
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, len_x):
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A)).contiguous()
        y = self.bn(x)
        y = self.relu(y)
        return y

class STGCN_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        A,
        adaptive=True,
        stride=1,
        dropout=0,
        residual=True,
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = GCN_unit(
            in_channels,
            out_channels,
            kernel_size[1],
            A,
            adaptive=adaptive,
        )
        if kernel_size[0] > 1:
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    (kernel_size[0], 1),
                    (stride, 1),
                    padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            )
        else:
            self.tcn = nn.Identity()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, len_x=None):
        res = self.residual(x)
        x = self.gcn(x, len_x)
        x = self.tcn(x) + res
        return self.relu(x)

class STGCNChain(nn.Sequential):
    def __init__(self, in_dim, block_args, kernel_size, A, adaptive):
        super(STGCNChain, self).__init__()
        last_dim = in_dim
        for i, [channel, depth] in enumerate(block_args):
            for j in range(depth):
                self.add_module(f'layer{i}_{j}', STGCN_block(last_dim, channel, kernel_size, A.clone(), adaptive))
                last_dim = channel

def get_stgcn_chain(in_dim, level, kernel_size, A, adaptive):
    if level == 'spatial':
        block_args = [[64,1], [128,1], [256,1]]
    elif level == 'temporal':
        block_args = [[256,3]]
    else:
        raise NotImplementedError
    return STGCNChain(in_dim, block_args, kernel_size, A, adaptive), block_args[-1][0]

# ---------------------------------------------------------------- pose branch of Uni_Sign (models.py, forward lines 216-267)
PARTS = {'body': slice(0, 9), 'left': slice(9, 30), 'right': slice(30, 51), 'face_all': slice(51, 69)}   # layout of our 69-point tensor

def _trunc_normal_(t, std=.02):
    nn.init.trunc_normal_(t, mean=0., std=std, a=-2., b=2.)

class UniPoseEncoder(nn.Module):
    """x: (B, T, 69, >=3) with channels x, y, conf in Uni-Sign's normalisation -> (B, T, 1024). Module names equal Uni_Sign's, so
    state-dict keys proj_linear.*, gcn_modules.*, fusion_gcn_modules.*, part_para load from a Uni-Sign checkpoint unchanged."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.modes = ['body', 'left', 'right', 'face_all']
        A = []
        self.proj_linear = nn.ModuleDict()
        for mode in self.modes:
            g = Graph(layout=f'{mode}', strategy='distance', max_hop=1)
            A.append(torch.tensor(g.A, dtype=torch.float32, requires_grad=False))
            self.proj_linear[mode] = nn.Linear(3, 64)
        self.gcn_modules = nn.ModuleDict(); self.fusion_gcn_modules = nn.ModuleDict()
        k = A[0].size(0)
        for i, mode in enumerate(self.modes):
            self.gcn_modules[mode], final_dim = get_stgcn_chain(64, 'spatial', (1, k), A[i].clone(), True)
            self.fusion_gcn_modules[mode], _ = get_stgcn_chain(final_dim, 'temporal', (5, k), A[i].clone(), True)
        self.gcn_modules['left'] = self.gcn_modules['right']
        self.fusion_gcn_modules['left'] = self.fusion_gcn_modules['right']
        self.proj_linear['left'] = self.proj_linear['right']
        self.part_para = nn.Parameter(torch.zeros(hidden_dim * len(self.modes)))
        for m in self.modules():                    # Uni_Sign._init_weights (Linear only; there is no LayerNorm in this branch)
            if isinstance(m, nn.Linear):
                _trunc_normal_(m.weight, .02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = x[..., :3]; feats = []; body_feat = None
        for part in self.modes:
            p = x[:, :, PARTS[part], :]
            proj = self.proj_linear[part](p).permute(0, 3, 1, 2)          # B,C,T,V
            g = self.gcn_modules[part](proj)
            if part == 'body': body_feat = g
            elif part == 'left': g = g + body_feat[..., -2][..., None].detach()
            elif part == 'right': g = g + body_feat[..., -1][..., None].detach()
            else: g = g + body_feat[..., 0][..., None].detach()
            g = self.fusion_gcn_modules[part](g)
            feats.append(g.mean(-1).transpose(1, 2))                      # B,T,C
        return torch.cat(feats, dim=-1) + self.part_para

def load_uni_checkpoint(enc, path):
    """Loads matching tensors (same key and shape) from a Uni-Sign checkpoint into enc. Returns a report dict; the caller decides whether it is good enough."""
    try: ck = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as e:
        print('weights_only load failed (%s); retrying with weights_only=False on the official checkpoint file' % type(e).__name__)
        ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    mine = enc.state_dict(); ok = {k: v for k, v in sd.items() if k in mine and tuple(v.shape) == tuple(mine[k].shape)}
    bad_shape = [k for k, v in sd.items() if k in mine and tuple(v.shape) != tuple(mine[k].shape)]
    enc.load_state_dict(ok, strict=False)
    enc_like = [k for k in sd if k.split('.')[0] in ('proj_linear', 'gcn_modules', 'fusion_gcn_modules', 'part_para')]
    return {'expected': len(mine), 'matched': len(ok), 'frac': len(ok) / len(mine), 'bad_shape': bad_shape[:10],
            'missing': [k for k in mine if k not in ok][:10], 'ckpt_tensors': len(sd), 'ckpt_encoder_like': len(enc_like), 'ckpt_other_examples': [k for k in sd if k not in mine][:5]}
