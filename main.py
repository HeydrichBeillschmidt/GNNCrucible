import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from load_graph import load_data
from models import Model
from sampler import ClusterSampler


def evaluate(model, g, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, g.ndata['feat'])
        logits = logits[mask]
        logits = logits.cpu().numpy()
        logits = np.argmax(logits, axis=-1)
        labels = labels[mask]
        labels = labels.cpu().numpy()
        return accuracy_score(labels, logits)


def train(model, g, loss_f, optimizer, cuda):
    if cuda:
        g = g.to(torch.cuda.current_device())
    model.train()
    # forward
    batch_labels = g.ndata['label']
    batch_feats = g.ndata['feat']
    pred = model(g, batch_feats)
    loss = loss_f(pred, batch_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def run(args):
    torch.manual_seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load and preprocess dataset
    g, n_classes = load_data(args)
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']

    train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)

    in_feats = g.ndata['feat'].shape[1]
    n_edges = g.number_of_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
            (n_edges, n_classes,
            n_train_samples,
            n_val_samples,
            n_test_samples))
    # metis only support int64 graph
    g = g.long()

    if n_train_samples >= 500:
        enable_cluster = True
        cluster_iterator = ClusterSampler(g, args.psize, args.batch_size, train_nid)
    else:
        enable_cluster = False

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        g = g.int().to(args.gpu)

    print('labels shape:', g.ndata['label'].shape)
    print("features shape, ", g.ndata['feat'].shape)

    model = Model(in_feats,
                  args.n_hidden,
                  n_classes,
                  n_layers=args.n_layers,
                  dropout=args.dropout)

    if cuda:
        model.cuda()

    # Loss function
    print('Using multi-class loss')
    loss_f = nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # set train_nids to cuda tensor
    if cuda:
        train_nid = torch.from_numpy(train_nid).cuda()
        print("current memory after model before training",
              torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)
    start_time = time.time()
    best_acc = -1

    for epoch in range(args.n_epochs):
        if enable_cluster:
            for j, cluster in enumerate(cluster_iterator):
                train(model, cluster, loss_f, optimizer, cuda)
        else:
            train(model, g, loss_f, optimizer, cuda)

        # evaluate
        if epoch % args.val_every == 0:
            val_acc = evaluate(model, g, labels, val_mask)
            if val_acc > best_acc:
                best_acc = val_acc
            print("In epoch {}, val acc {:.4f}(best {:.4f})".format(epoch, val_acc, best_acc))

    end_time = time.time()
    print('training using time {:.4f}'.format(end_time-start_time))

    # test
    test_acc = evaluate(model, g, labels, test_mask)
    print("Test acc {:.4f}".format(test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="batch size")
    parser.add_argument("--psize", type=int, default=16,
                        help="partition number")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="number of hidden gcn layers")
    parser.add_argument("--val-every", type=int, default=5,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--rnd-seed", type=int, default=3,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")

    args = parser.parse_args()

    run(args)
