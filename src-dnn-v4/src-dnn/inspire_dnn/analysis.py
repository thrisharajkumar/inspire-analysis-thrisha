# Analysis helpers for "looking inside the model" (v4, Part 13). Numpy / scikit-learn /
# matplotlib only -- no PyTorch, no dataset globals -- so every function is unit-tested on
# random arrays (tests/test_analysis.py) and inlined into the notebooks unchanged.
#
# What lives here:
#   * the four patient groups James asked for (emergency / scheduled  x  died / survived)
#   * metrics per group (AUROC, AUPRC with bootstrap CIs, Brier)
#   * squashing embeddings onto a 2-D map (PCA, t-SNE, UMAP if installed)
#   * "do deaths sit near deaths?" -- k-nearest-neighbour death enrichment
#   * unsupervised clustering of the embeddings + a plain-English profile of each cluster
#   * cosine-similarity matrices and a SHAP-style beeswarm plot
import numpy as np
import pandas as pd

GROUP_ORDER = ["Scheduled - survived", "Emergency - survived", "Scheduled - died", "Emergency - died"]
GROUP_COLORS = {"Scheduled - survived": "#6baed6", "Emergency - survived": "#fd8d3c",
                "Scheduled - died": "#08519c", "Emergency - died": "#d94801"}


def group_labels(emop, died):
    """Emergency/scheduled x died/survived label per patient (NaN emop -> 'Scheduled')."""
    emop = np.asarray(emop, dtype=float)
    died = np.asarray(died, dtype=float)
    em = np.where(np.isnan(emop), 0.0, emop) > 0.5
    out = np.empty(len(died), dtype=object)
    for i in range(len(died)):
        out[i] = f"{'Emergency' if em[i] else 'Scheduled'} - {'died' if died[i] > 0.5 else 'survived'}"
    return out


def _safe_metric(fn, y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(fn(y, p))


def bootstrap_ci(y, p, fn, n=200, seed=0, alpha=0.05):
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2 or n <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def metrics_by_group(y, p, groups, n_boot=200, seed=0, threshold=None):
    """One row per group (+ 'All'): n, deaths, death rate, AUROC/AUPRC (+95% CI), Brier,
    mean predicted risk, and -- if a threshold is given -- recall (deaths caught) and precision."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float); groups = np.asarray(groups, dtype=object)
    rows = []
    for g in ["All"] + [x for x in pd.unique(groups)]:
        sel = np.ones(len(y), bool) if g == "All" else (groups == g)
        yy, pp = y[sel], p[sel]
        if len(yy) == 0:
            continue
        row = {"group": g, "n": int(len(yy)), "deaths": int(yy.sum()), "death_rate": float(yy.mean()),
               "mean_predicted_risk": float(pp.mean()),
               "auroc": _safe_metric(roc_auc_score, yy, pp), "auprc": _safe_metric(average_precision_score, yy, pp),
               "brier": float(brier_score_loss(yy, pp)) if len(yy) else np.nan}
        lo, hi = bootstrap_ci(yy, pp, roc_auc_score, n_boot, seed); row["auroc_ci"] = f"{lo:.3f}-{hi:.3f}"
        lo, hi = bootstrap_ci(yy, pp, average_precision_score, n_boot, seed); row["auprc_ci"] = f"{lo:.3f}-{hi:.3f}"
        if threshold is not None:
            pred = pp >= threshold
            tp = int((pred & (yy == 1)).sum()); fn = int((~pred & (yy == 1)).sum()); fp = int((pred & (yy == 0)).sum())
            row["recall_at_threshold"] = tp / max(tp + fn, 1)
            row["precision_at_threshold"] = tp / max(tp + fp, 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def project_2d(X, method="tsne", seed=0, perplexity=30.0):
    """Squash an [n, d] embedding onto 2-D. method: 'pca' | 'tsne' | 'umap'.
    Returns None for 'umap' if umap-learn is not installed (reported, never faked)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(StandardScaler().fit_transform(X))
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        Xr = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=seed).fit_transform(X) if X.shape[1] > 50 else X
        perp = float(min(perplexity, max(5.0, (len(X) - 1) / 3.0)))
        return TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(Xr)
    if method == "umap":
        try:
            import umap
        except ImportError:
            return None
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
    raise ValueError(method)


def knn_death_enrichment(X, died, k=15):
    """Among each patient's k nearest neighbours IN THE FULL EMBEDDING (not the 2-D map),
    what fraction died? Reported for died vs survived patients, and as a ratio to the base
    rate. Ratio >> 1 for died patients = deaths cluster together = the embedding has learned
    something about risk. ~1 = deaths are scattered at random."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    X = np.nan_to_num(StandardScaler().fit_transform(np.asarray(X, dtype=float)))
    died = np.asarray(died, dtype=float)
    k = int(min(k, len(X) - 1))
    if k < 1 or died.sum() == 0:
        return {"k": k, "base_rate": float(died.mean()), "died_neighbour_rate_for_died": np.nan,
                "died_neighbour_rate_for_survived": np.nan, "enrichment_ratio": np.nan}
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, ind = nn_.kneighbors(X)
    frac = died[ind[:, 1:]].mean(axis=1)
    base = float(died.mean())
    rd = float(frac[died == 1].mean())
    rs = float(frac[died == 0].mean()) if (died == 0).any() else np.nan
    return {"k": k, "base_rate": base, "died_neighbour_rate_for_died": rd,
            "died_neighbour_rate_for_survived": rs, "enrichment_ratio": rd / max(base, 1e-12)}


def cluster_embeddings(X, k_range=(3, 4, 5, 6, 7, 8), seed=0, max_silhouette_n=5000):
    """K-means over the standardised embedding for each k; picks the k with the best
    silhouette (how cleanly separated the clusters are, -1..1). Returns (labels, table)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    X = np.nan_to_num(StandardScaler().fit_transform(np.asarray(X, dtype=float)))
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X), size=min(max_silhouette_n, len(X)), replace=False)
    best, rows = None, []
    for k in k_range:
        if k >= len(X):
            continue
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        if len(np.unique(lab[sub])) < 2:
            continue
        sil = float(silhouette_score(X[sub], lab[sub]))
        rows.append({"k": k, "silhouette": sil})
        if best is None or sil > best[0]:
            best = (sil, k, lab)
    table = pd.DataFrame(rows)
    return (best[2] if best else np.zeros(len(X), int)), table


def cluster_profile(clusters, died, emop=None, extra=None):
    """Plain-English profile of each cluster: size, death rate, emergency share, and the
    most common value / mean of any extra columns (e.g. department, ASA, top system)."""
    df = pd.DataFrame({"cluster": np.asarray(clusters), "died": np.asarray(died, dtype=float)})
    if emop is not None:
        df["emergency"] = np.asarray(emop, dtype=float)
    for name, vals in (extra or {}).items():
        df[name] = np.asarray(vals)
    agg = {"died": ["size", "sum", "mean"]}
    if emop is not None:
        agg["emergency"] = "mean"
    out = df.groupby("cluster").agg(agg)
    out.columns = ["n_patients", "deaths", "death_rate"] + (["emergency_share"] if emop is not None else [])
    for name in (extra or {}):
        col = df[name]
        if pd.api.types.is_numeric_dtype(col):
            out[f"mean_{name}"] = df.groupby("cluster")[name].mean()
        else:
            out[f"most_common_{name}"] = df.groupby("cluster")[name].agg(lambda s: s.value_counts().index[0] if len(s.dropna()) else "")
    out["death_rate_vs_overall"] = out["death_rate"] / max(df["died"].mean(), 1e-12)
    return out.sort_values("death_rate", ascending=False)


def cosine_similarity_matrix(X):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(n, 1e-12, None)
    return Xn @ Xn.T


def plot_embedding_map(ax, Z, groups, title="", order=GROUP_ORDER, colors=GROUP_COLORS, s_small=6, s_big=22):
    """Scatter a 2-D map: survivors first (small, faint), deaths on top (bigger), so the rare
    deaths are never hidden under thousands of survivors."""
    groups = np.asarray(groups, dtype=object)
    for g in order:
        sel = groups == g
        if not sel.any():
            continue
        died = g.endswith("died")
        ax.scatter(Z[sel, 0], Z[sel, 1], s=s_big if died else s_small, alpha=0.9 if died else 0.5,
                   c=colors.get(g, "#888888"), label=f"{g} (n={int(sel.sum())})",
                   edgecolors="black" if died else "none", linewidths=0.3 if died else 0)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def plot_categorical_map(ax, Z, values, title="", max_levels=10, s=6):
    values = pd.Series(np.asarray(values, dtype=object)).fillna("missing").astype(str)
    top = values.value_counts().index[:max_levels]
    values = values.where(values.isin(top), "other")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10" if len(top) <= 10 else "tab20")
    for i, lev in enumerate(values.value_counts().index):
        sel = (values == lev).values
        ax.scatter(Z[sel, 0], Z[sel, 1], s=s, alpha=0.6, color=cmap(i % cmap.N), label=f"{lev} ({int(sel.sum())})")
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def beeswarm(ax, attributions, values, names, top=10, title=""):
    """SHAP-style summary: one row per feature (most important at the top), one dot per
    patient, x = attribution (+ pushes towards 'died'), colour = the feature's own value
    (blue low -> red high, after standardisation)."""
    A = np.asarray(attributions, dtype=float); V = np.asarray(values, dtype=float)
    order = np.argsort(-np.nanmean(np.abs(A), axis=0))[:top][::-1]
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        a = A[:, j]; v = V[:, j]
        lo, hi = np.nanpercentile(v, 5) if np.isfinite(v).any() else 0, np.nanpercentile(v, 95) if np.isfinite(v).any() else 1
        c = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
        y = row + rng.uniform(-0.3, 0.3, size=len(a))
        ax.scatter(a, y, c=np.nan_to_num(c, nan=0.5), cmap="coolwarm", s=5, alpha=0.7, vmin=0, vmax=1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[j] for j in order], fontsize=8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("attribution (points of this term's risk)", fontsize=8); ax.set_title(title, fontsize=10)
    return [names[j] for j in order[::-1]]


def mean_abs_importance(attributions, names):
    A = np.asarray(attributions, dtype=float)
    return pd.Series(np.nanmean(np.abs(A), axis=0), index=list(names)).sort_values(ascending=False)
