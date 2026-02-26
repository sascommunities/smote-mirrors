# Copyright © 2026, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from tqdm.notebook import tqdm

# modelling
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# generators
from imblearn.over_sampling import SMOTE


def featurize_X(X):
    features = []
    for i in range(X.shape[1]):
        col_data = X[:, i]
        features.extend([np.min(col_data), np.quantile(col_data, 0.25), np.mean(col_data),
                         np.median(col_data), np.quantile(col_data, 0.75), np.max(col_data)])
    
    # correalations
    corrs = np.cov(X.T)
    idx = np.tril(np.ones(corrs.shape)).astype(bool)
    features.extend(corrs[idx])

    return features


def membership_attack(X, y, minority_target_idx, n_train, n_test, attack_mode=None, clf_kwargs=None, gen_kwargs=None):
    assert attack_mode in ("clf", "aug+clf", "gen")
    
    n_majority, n_minority = (y == 0).sum(), (y == 1).sum()
    X_real_majority = X[(y==0)]
    X_real_minority = X[(y==1)]

    target = X_real_minority[minority_target_idx].reshape(1, -1)

    X_in = np.concatenate((X_real_majority, X_real_minority))
    y_in = np.concatenate((np.zeros(n_majority), np.ones(n_minority))).astype(int)
    X_out = np.concatenate((X_real_majority, X_real_minority[:minority_target_idx], X_real_minority[minority_target_idx+1:]))
    y_out = np.concatenate((np.zeros(n_majority), np.ones(n_minority - 1))).astype(int)

    n_all = n_train + n_test
    if "clf" in attack_mode:
        in_feats, out_feats = np.zeros([n_all]), np.zeros([n_all])
    else:
        n_feats = len(featurize_X(X))
        in_feats, out_feats = np.zeros([n_all, n_feats]), np.zeros([n_all, n_feats])

    for j in range(n_all):
        # run MIA on a classifier trained on real data -- train shadow classifiers
        if attack_mode == "clf":
            clf_in = MLPClassifier(**clf_kwargs)
            clf_in.fit(X_in, y_in)
            in_feats[j] = clf_in.predict_proba(target)[0, 1]

            clf_out = MLPClassifier(**clf_kwargs)
            clf_out.fit(X_out, y_out)
            out_feats[j] = clf_out.predict_proba(target)[0, 1]

        # run MIA on a classifier trained on augmented data -- train shadow classifiers
        elif attack_mode == "aug+clf":
            generator_in = SMOTE(**gen_kwargs)
            X_aug_in, y_aug_in = generator_in.fit_resample(X_in, y_in)
            clf_in = MLPClassifier(**clf_kwargs)
            clf_in.fit(X_aug_in, y_aug_in)
            in_feats[j] = clf_in.predict_proba(target)[0, 1]

            generator_out = SMOTE(**gen_kwargs)
            X_aug_out, y_aug_out = generator_out.fit_resample(X_out, y_out)
            clf_out = MLPClassifier(**clf_kwargs)
            clf_out.fit(X_aug_out, y_aug_out)
            out_feats[j] = clf_out.predict_proba(target)[0, 1]

        # run MIA on generator/synthetic data -- train shadow generators and generate synthetic data
        else:
            generator_in = SMOTE(**gen_kwargs)
            X_aug_in, _ = generator_in.fit_resample(X_in, y_in)
            X_synth_in = X_aug_in[len(X_in):]
            in_feats[j] = featurize_X(X_synth_in)

            generator_out = SMOTE(**gen_kwargs)
            X_aug_out, _ = generator_out.fit_resample(X_out, y_out)
            # discard 1 synthetic point so we have balanced in/out datasets
            X_synth_out = X_aug_out[len(X_out)+1:]
            out_feats[j] = featurize_X(X_synth_out)

    target_test_labels = np.concatenate((np.ones(n_test), np.zeros(n_test))).astype(int)
    # directly use the shadow classifiers predictions
    if "clf" in attack_mode:
        target_test_preds = np.concatenate((in_feats[n_train: ], out_feats[n_train: ]))

    # train a meta classifier on synthetic data from the shadow generators and use its predictions
    else:
        target_train_feats = np.concatenate((in_feats[: n_train], out_feats[: n_train]))
        target_train_labels = np.concatenate((np.ones(n_train), np.zeros(n_train))).astype(int)
        target_test_feats = np.concatenate((in_feats[n_train: ], out_feats[n_train: ]))
    
        clf = RandomForestClassifier()
        clf.fit(target_train_feats, target_train_labels)
        target_test_preds = clf.predict_proba(target_test_feats)[:, 1]

    return target_test_labels, target_test_preds