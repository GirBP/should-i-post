"""Two evaluation protocols, applied to every experiment.

  temporal : train < valid < test by create_date (deployment-realistic).
  LOCO     : leave-one-creator-out via GroupKFold on author_id (cross-creator
             generalisation — the honest test of "knowledge from other creators").
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from . import config as C


def temporal_masks(df: pd.DataFrame):
    s = df["split_temporal"].values
    return (s == "train"), (s == "valid"), (s == "test")


def loco_folds(df: pd.DataFrame, n_splits: int = C.LOCO_FOLDS):
    """Yield (train_idx, test_idx) with disjoint authors across folds."""
    groups = df["author_id"].values
    y = np.zeros(len(df))
    return list(GroupKFold(n_splits=n_splits).split(df, y, groups))
