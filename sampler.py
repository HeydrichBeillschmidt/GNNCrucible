import dgl
import dgl.backend as F
from dgl.transform import metis_partition
from dgl.dataloading import BlockSampler
import numpy as np
import random


class ClusterSampler(object):
    def __init__(self, graph,
                 n_partition: int,
                 batch_size: int,
                 seed_nid):
        super(ClusterSampler, self).__init__()
        self.graph = graph.subgraph(seed_nid)
        self.n_part = n_partition
        self.batch_size = batch_size
        self.prt_list = []
        self.get_partition_list()
        random.shuffle(self.prt_list)
        self.max = int(n_partition // batch_size)

    def get_partition_list(self):
        sub_gs = metis_partition(self.graph, self.n_part)
        for _, sub_g in sub_gs.items():
            n_ids = sub_g.ndata[dgl.NID]
            n_ids = F.asnumpy(n_ids)
            self.prt_list.append(n_ids)

    def get_subgraph(self, i: int):
        batch_idx = [self.prt_list[s] for s in range(
            i * self.batch_size, (i+1) * self.batch_size) if s < self.n_part]
        return self.graph.subgraph(np.concatenate(
            batch_idx).reshape(-1).astype(np.int64))

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.max:
            ans = self.get_subgraph(self.n)
            self.n += 1
            return ans
        else:
            random.shuffle(self.prt_list)
            raise StopIteration

    def __len__(self):
        return self.max
