import torch as th

datasets_dgl = ['cora', 'citeseer', 'pubmed', 'reddit']
datasets_ogb = ['ogbn-products', 'ogbn-proteins', 'ogbn-arxiv', 'ogbn-mag']


def load_data(args):
    if args.dataset in datasets_dgl:
        from dgl.data import load_data
        dataset = load_data(args)
        return dataset[0], dataset.num_classes
    elif args.dataset in datasets_ogb:
        return load_ogb(args.dataset)
    else:
        raise Exception('Unknown dataset: {}'.format(args.dataset))


def load_ogb(name):
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)
    data = DglNodePropPredDataset(name=name)
    print('finish loading', name)
    split_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    # graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['label'] = labels
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = split_idx['train'], split_idx['valid'], split_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    return graph, num_labels
