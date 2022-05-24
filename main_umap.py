import argparse
import itertools
import sys
import time

from pathlib import Path
from numpy import genfromtxt
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
from umap import UMAP
from metric_mean_dist import generate_raport


def plot_umap(embedding, y, title, filename, size=15, y_names=None):
    plt.figure(figsize=(16, 10))
    labels = np.unique(y)
    for i in labels:
        mask = (y == i)
        if y_names:
            plt.scatter(embedding[mask, 0], embedding[mask, 1], label=y_names[i], s=size)
        else:
            plt.scatter(embedding[mask, 0], embedding[mask, 1], label=i, s=size)

    plt.title(title)
    legend = plt.legend()
    for i in range(labels.shape[0]):
        legend.legendHandles[i]._sizes = [30]

    plt.savefig(f'visualization/umap/{filename}')


def read_data(dataset='fmnist'):

    if dataset in ('fmnist', 'smallnorb', 'reuters'):
        return genfromtxt(f'datasets/{dataset}.csv', delimiter=',')

    else:
        print('Dataset does not exist!')
        raise IOError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset choice")
    parser.add_argument("dataset", type=str, help="choose dataset to make reduction")
    args = parser.parse_args()
    Path("visualization/umap").mkdir(parents=True, exist_ok=True)
    Path("manifolds/umap").mkdir(parents=True, exist_ok=True)
    Path("metrics/umap").mkdir(parents=True, exist_ok=True)

    data = read_data(args.dataset)
    labels = data[:, -1]
    labels = labels.astype(int)
    data = data[:, :-1]

    # preprocessing
    n_neighbors = [3, 5, 10, 20, 100]
    min_dists = [0.0, 0.1, 0.25, 0.5]

    parameters_grid = itertools.product(n_neighbors, min_dists)
    for parameters in parameters_grid:
        nn, min_dist = parameters
        t1 = time.time()
        reducer = UMAP(n_components=2, n_jobs=8, min_dist=min_dist, n_neighbors=nn)
        manifold = reducer.fit_transform(data)
        filename = f'{args.dataset}_TSNE_n_neighbors-{nn}_min_dist-{int(min_dist*100)}'
        print(f'N_neighbors {nn} | min_dist {min_dist} | Time {time.time() - t1}')
        manifold.dump(f'manifolds/umap/{filename}')

        if args.dataset == 'fmnist':
            plot_umap(
                embedding=manifold, y=labels, filename=filename, size=10,
                title=f'Dataset {args.dataset} | N_neighbors {nn} | Min_dist {min_dist}',
                y_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                         'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            )
        else:
            plot_umap(
                embedding=manifold, y=labels, filename=filename, size=10,
                title=f'Dataset {args.dataset} | N_neighbors {nn} | Min_dist {min_dist}'
            )

        generate_raport(manifold, labels, filename=filename, method='umap')
