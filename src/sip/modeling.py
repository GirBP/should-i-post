"""Model factory, probability calibration, and split-conformal abstention."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from . import config as C


def make_model(name: str):
    """Return an unfitted sklearn-compatible classifier with predict_proba."""
    name = name.lower()
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(with_mean=True),
                             LogisticRegression(max_iter=1000, C=1.0))
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, random_state=C.SEED)
    if name == "mlp":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                           max_iter=200, random_state=C.SEED))
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=C.SEED)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
            verbose=False, random_seed=C.SEED, allow_writing_files=False)
    raise ValueError(f"unknown model {name}")


# ----------------------------------------------------------------- calibration
class Calibrated:
    """Wrap a fitted model with isotonic or sigmoid (Platt) calibration learned
    on a held-out calibration set. Keeps base ranking, fixes probabilities."""

    def __init__(self, base, method="isotonic"):
        self.base = base
        self.method = method
        self._cal = None

    def fit_calibration(self, X_cal, y_cal):
        p = self.base.predict_proba(X_cal)[:, 1]
        y = np.asarray(y_cal).astype(int)
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self._cal = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        else:
            from sklearn.linear_model import LogisticRegression
            self._cal = LogisticRegression(max_iter=1000).fit(p.reshape(-1, 1), y)
        return self

    def predict_proba_pos(self, X):
        p = self.base.predict_proba(X)[:, 1]
        if self._cal is None:
            return p
        if self.method == "isotonic":
            return self._cal.predict(p)
        return self._cal.predict_proba(p.reshape(-1, 1))[:, 1]


# ------------------------------------------------------- split-conformal abstain
class ConformalAbstention:
    """Class-conditional (Mondrian) split-conformal prediction sets.

    For target miscoverage alpha, include class k in the prediction set when the
    candidate's score for k is at least the (1-alpha) calibrated threshold for k.
    Singleton set -> confident {Post/Do-not-post}; otherwise -> Unsure (abstain).
    Honest: coverage guarantee holds marginally per class on exchangeable data.
    """

    def __init__(self, alpha=C.CONFORMAL_ALPHA):
        self.alpha = alpha
        self.q = {}

    def fit(self, p_cal, y_cal):
        p = np.asarray(p_cal, dtype=float)
        y = np.asarray(y_cal).astype(int)
        proba = np.c_[1 - p, p]                      # [P(0), P(1)]
        for k in (0, 1):
            mk = y == k
            if mk.sum() == 0:
                self.q[k] = 1.0; continue
            scores = 1.0 - proba[mk, k]              # nonconformity for true class k
            n = mk.sum()
            level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
            self.q[k] = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_set(self, p):
        p = np.asarray(p, dtype=float)
        proba = np.c_[1 - p, p]
        inc0 = (1.0 - proba[:, 0]) <= self.q.get(0, 1.0)
        inc1 = (1.0 - proba[:, 1]) <= self.q.get(1, 1.0)
        return inc0.astype(int) + 2 * inc1.astype(int)   # 1={0}, 2={1}, 3={0,1}, 0={}

    def decisions(self, p):
        s = self.predict_set(p)
        out = np.full(len(s), "Unsure", dtype=object)
        out[s == 2] = "Post"
        out[s == 1] = "Do not post"
        return out
