# Copyright © 2026, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd


def check_collinear(a , b, c, line_eps=1e-12):
    # check if a->b and a->c are collinear
    ab = b - a
    ac = c - a
    ab_norm = np.linalg.norm(ab)
    ac_norm = np.linalg.norm(ac)
    if ab_norm == 0 or ac_norm == 0:
        return False
    
    cos_theta = np.dot(ab, ac) / (ab_norm * ac_norm)
    return np.abs(np.abs(cos_theta) - 1) < line_eps


def check_midpoint(a, b, c, line_eps=1e-12):
    # we assume a->b and a->c are collinear
    # check if c ia a midpoint on a-b
    ab = b - a
    ac = c - a

    proj = np.dot(ab, ac) / np.dot(ab, ab)
    return -line_eps < proj < 1 + line_eps


def find_intersection_point(points_i, points_j, line_eps=1e-12):
    p1 = np.mean(points_i, axis=0)
    v1 = points_i[0] - p1

    p2 = np.mean(points_j, axis=0)
    v2 = points_j[0] - p2

    # solve for scalars a and b such that:
    # p1 + a*v1 = p2 + b*v2 => a*v1 - b*v2 = p2 - p1
    A = np.column_stack((v1, -v2))
    b = p2 - p1

    try:
        a_, b_ = np.linalg.lstsq(A, b, rcond=None)[0]
        # compute the potential intersection point from both lines
        point_on_line_i = p1 + a_ * v1
        point_on_line_j = p2 + b_ * v2
        # check if the points are equal
        if np.allclose(point_on_line_i, point_on_line_j, atol=line_eps):
            return True, point_on_line_i
        else:
            return False, None
    except np.linalg.LinAlgError:
        return False, None


def calculate_scores(X, y, predictions, exact_match=True, line_eps=1e-12):
    if len(predictions) == 0:
        return (0, 0, 0)

    X_minority = X[y==1]
    
    # drop duplicates in X_minority -- remove exact duplicates
    X_minority_unique = np.unique(X_minority, axis=0)
    # NOTE: we already checked for collinearity in minority records for all datasets -- there are none!!!
    
    # drop duplicates in predictions -- remove approximate duplicates
    duplicates = np.all(np.isclose(predictions[:, None], predictions[None, :], atol=line_eps), axis=2)
    np.fill_diagonal(duplicates, False)
    predictions_unique = predictions[~np.any(np.triu(duplicates), axis=1)]

    if exact_match:
        # exact match
        true_positives = pd.DataFrame(predictions_unique).apply(tuple, 1).isin(pd.DataFrame(X_minority_unique).apply(tuple, 1)).sum()
    else:
        # approximate match
        matches = np.all(np.isclose(predictions_unique[:, None, :], X_minority_unique[None, :, :], atol=line_eps), axis=2)
        true_positives = np.sum(np.any(matches, axis=1))

    precision = true_positives / len(predictions)
    recall = true_positives / len(X_minority_unique)
    f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)
