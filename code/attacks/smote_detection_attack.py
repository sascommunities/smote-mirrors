# Copyright © 2026, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from collections import deque
from scipy.spatial import ConvexHull, QhullError
from sklearn.neighbors import NearestNeighbors

# custom functions
from attacks import check_collinear, check_midpoint


def smote_detection_attack(X_augmented, y_augmented, use_hull=True, k=5, r=5, nbs_multiplte=2.5, line_eps=1e-12, return_details=False):
    # we assume the minority class has a label 1
    X_minority = X_augmented[y_augmented == 1]
    N, d = X_minority.shape

    hull_set = set()
    if use_hull:
        try:
            hull = ConvexHull(X_minority)
            hull_set = set(hull.vertices)
            # print("ConvexHull extracted")
        except QhullError:
            for col in range(d):
                col_vals = X_minority[:, col]
                min_indices = np.where(col_vals == np.min(col_vals))[0]
                max_indices = np.where(col_vals == np.max(col_vals))[0]
                if len(min_indices) == 1:
                    hull_set.add(min_indices[0])
                if len(max_indices) == 1:
                    hull_set.add(max_indices[0])

    candidate_mask = np.ones(N, dtype=bool)
    
    k_ = k
    k_nbs = min(int(nbs_multiplte * k_ * r + 1), N)
    tree = NearestNeighbors(n_neighbors=k_nbs).fit(X_minority)

    queue = deque(list(hull_set))
    if not queue or not use_hull:
        queue = deque(np.random.choice(N, N//10))
    
    history = []
    iter = 0

    while queue and iter < 10:
        next_queue = deque()
        visited = set()

        while queue:
            i = queue.popleft()
            if i in visited or not candidate_mask[i]:
                continue
            visited.add(i)
            history.append([i, {}])
            
            nbs = tree.kneighbors(X_minority[i].reshape(1, -1), return_distance=False)[0][1:]
            nbs = [j for j in nbs if candidate_mask[j] or j in hull_set]
            history[-1][1]["nbs"] = nbs

            to_remove = set()

            # check triplets (i, j, k) for midpoiint pruning
            for idx_j, j in enumerate(nbs):
                for k in nbs[idx_j + 1:]:
                    a, b, c = X_minority[i], X_minority[j], X_minority[k]
                    triples = [(a, b, c, k), # is c midpoint on a-b
                               (c, a, b, j), # is b midpoint on c-a
                               (b, c, a, i), # is a midpoint on b-c
                            ]

                    if check_collinear(a, b, c, line_eps):
                        for u, v, w, w_idx in triples:
                            if w_idx not in hull_set and check_midpoint(u, v, w, line_eps):
                                to_remove.add(w_idx)
                                break
            
            if to_remove:
                candidate_mask[list(to_remove)] = False
                history[-1][1]["to_remove"] = to_remove
                
                for nb in nbs:
                    if candidate_mask[nb] and nb not in hull_set:
                        next_queue.append(nb)
            history[-1][1]["candidate_mask"] = candidate_mask.copy()

        queue = next_queue
        iter += 1

    if return_details:
        return X_minority[candidate_mask], history
    return X_minority[candidate_mask]
  