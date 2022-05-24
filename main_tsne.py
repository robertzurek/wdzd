import argparse
import itertools
import sys
import time

from pathlib import Path
from numpy import genfromtxt
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
from metric_mean_dist import generate_raport

def plot_tsne(embedding, y, title, filename, size=15, y_names=None):

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

    plt.savefig(f'visualization/tsne/{filename}')


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
    Path("visualization/tsne").mkdir(parents=True, exist_ok=True)
    Path("manifolds/tsne").mkdir(parents=True, exist_ok=True)
    Path("metrics/tsne").mkdir(parents=True, exist_ok=True)

    data = read_data(args.dataset)
    labels = data[:, -1]
    labels = labels.astype(int)
    data = data[:, :-1]

    # preprocessing
    perplexity_parameters = [2, 5, 10, 20, 50, 100]

    for perplexity in perplexity_parameters:
        for n_run in range(1, 3):
            t1 = time.time()
            reducer = TSNE(n_components=2, n_jobs=8, perplexity=perplexity)
            manifold = reducer.fit_transform(data)
            filename = f'{args.dataset}_TSNE_{n_run}_perplexity-{perplexity}'
            print(f'Perplexity {perplexity} | n_run {n_run} | Time {time.time() - t1}')
            manifold.dump(f'manifolds/tsne/{filename}')

            if args.dataset == 'fmnist':
                plot_tsne(
                    embedding=manifold, y=labels, filename=filename, size=10,
                    title=f'Dataset {args.dataset} | Perplexity {perplexity}',
                    y_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                             'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                )
            else:
                plot_tsne(
                    embedding=manifold, y=labels, filename=filename, size=10,
                    title=f'Dataset {args.dataset} | Perplexity {perplexity}'
                )

            generate_raport(manifold, labels, filename=filename, method='tsne')
