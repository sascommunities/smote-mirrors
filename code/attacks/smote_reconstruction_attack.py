# Copyright © 2026, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from sklearn.neighbors import NearestNeighbors

# custom functions
from attacks import check_collinear, find_intersection_point


def find_line(i, nbs, X_minority, line_eps=1e-12):
    for idx_j, j in enumerate(nbs):
        for k in nbs[idx_j + 1:]:
            a, b, c = X_minority[i], X_minority[j], X_minority[k]
            if check_collinear(a, b, c, line_eps):
                return j, k, a, b, c
    
    return None


def smote_reconstruction_attack(X_synthetic, y_synthetic, k=5, r=5, nbs_multiplte=2, inter_multiple=3, line_eps=1e-12, return_details=False):
    # we assume the minority class has a label 1
    X_minority = X_synthetic[y_synthetic == 1]
    N, d = X_minority.shape
    
    lines, lines_midpoints = [], []
    
    k_ = k
    k_nbs = min(int(nbs_multiplte * k_ * r + 1), N)
    tree = NearestNeighbors(n_neighbors=k_nbs).fit(X_minority)

    visited = set()

    for i in range(N):
        if i in visited:
            continue
        visited.add(i)
        
        nbs = tree.kneighbors(X_minority[i].reshape(1, -1), return_distance=False)[0][1:]
        nbs = [j for j in nbs if j not in visited]

        # define the line i belongs to by finding 2 neighboring points on the same line
        line_result = find_line(i, nbs, X_minority, line_eps)
        if line_result:
            j, k, a, b, _ = line_result
            lines.append(set([i, j, k]))
            visited.update([j, k])

            # find all remaining neighbors on the line
            for l in nbs:
                if l in [i, j, k]:
                    continue
                
                c = X_minority[l]
                if check_collinear(a, b, c, line_eps):
                    lines[-1].add(l)
                    visited.add(l)

            line_points = X_minority[list(lines[-1])]
            lines_midpoints.append(np.mean(line_points, axis=0))

    # iterate over lines
    lines_midpoints = np.array(lines_midpoints)
    if len(lines_midpoints) == 0:
        return np.array([])

    reconstructed_points, lines_intersections = [], []

    k_nbs_midpoints = min(int(nbs_multiplte * k_ + 1), len(lines_midpoints))
    tree_midpoints = NearestNeighbors(n_neighbors=k_nbs_midpoints).fit(lines_midpoints)

    visited = set()

    for i, midpoint in enumerate(lines_midpoints):
        nbs = tree_midpoints.kneighbors(midpoint.reshape(1, -1), return_distance=False)[0][1:]

        for j in nbs:
            ord_tup = (min(i,j), max(i,j))
            if ord_tup in visited:
                continue
            visited.add(ord_tup)

            points_line_i = X_minority[list(lines[i])]
            points_line_j = X_minority[list(lines[j])]

            intersect_flag, point = find_intersection_point(points_line_i, points_line_j, line_eps)
            if intersect_flag:
                for idx, existing_point in enumerate(reconstructed_points):
                    if np.allclose(point, existing_point, atol=line_eps):
                        lines_intersections[idx].add(ord_tup)
                        break
                else:
                    reconstructed_points.append(point)
                    lines_intersections.append(set([ord_tup]))

    # filter intersection point with fewer than 3 lines
    idx = [i for i, intrs in enumerate(lines_intersections) if len(intrs) >= inter_multiple]
    lines_intersections = [lines_intersections[i] for i in idx]
    reconstructed_points = np.array([reconstructed_points[i] for i in idx])

    if len(reconstructed_points) == 0:
        return np.array([])

    if return_details:
        return reconstructed_points, (lines, lines_midpoints, lines_intersections)
    return reconstructed_points
