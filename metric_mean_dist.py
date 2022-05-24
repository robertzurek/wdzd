from pathlib import Path

from scipy.spatial.distance import cdist
import numpy as np
import itertools


def generate_raport(embedding, y_target, method, filename):
    # dist for same classes
    log = []
    classes = np.unique(y_target)
    all_dist = []
    for i in classes:
        ix = y_target == i
        same_class_points = embedding[ix, :]
        points_combinations = itertools.combinations(same_class_points, 2)
        euclidean_dist = 0
        class_dist = []
        for comb in points_combinations:
            A, B = comb
            A = np.reshape(A, (1, 2))
            B = np.reshape(B, (1, 2))
            class_dist.append(cdist(A, B, metric='euclidean')[0, 0])
        print(f'Mean dist for class {i}: {np.mean(class_dist):.5f}')
        log.append(f'Mean dist for class {i}: {np.mean(class_dist):.5f}')
        all_dist += class_dist
    same_class_mean = np.mean(all_dist)
    print(f'Mean distance for same classes: {same_class_mean:.5f}')
    log.append(f'Mean distance for same classes: {same_class_mean:.5f}')

    # calculate dist for different classes
    total_dist = 0
    n_combinations = 0
    for i in classes:
        ix = y_target == i
        same_class_points = embedding[ix, :]
        other_points = embedding[np.invert(ix), :]
        for point in same_class_points:
            temp = other_points - point
            temp = np.power(temp, 2)
            total_dist += np.sum(np.sqrt(np.sum(temp, axis=1)))
            n_combinations += other_points.shape[0]
    diff_class_mean = np.mean(total_dist / n_combinations)
    print(f'Mean distance for different classes: {diff_class_mean:.5f}')
    log.append(f'Mean distance for different classes: {diff_class_mean:.5f}')
    print(f'Total ratio: {same_class_mean / diff_class_mean:.5f}')
    log.append(f'Total ratio: {same_class_mean / diff_class_mean:.5f}')

    # save to file
    Path(f"metrics/{method}").mkdir(parents=True, exist_ok=True)
    with open(f'metrics/{method}/{filename}.txt', mode='w', encoding='utf-8') as f:
        for line in log:
            f.write(line)
