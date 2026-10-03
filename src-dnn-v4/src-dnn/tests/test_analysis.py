# v4: numpy-only analysis helpers (Part 13).
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from inspire_dnn.analysis import (group_labels, metrics_by_group, project_2d, knn_death_enrichment,
                                  cluster_embeddings, cluster_profile, cosine_similarity_matrix,
                                  plot_embedding_map, beeswarm, GROUP_ORDER)

rng = np.random.default_rng(0)
N = 300
died = (rng.random(N) < 0.1).astype(float)
emop = (rng.random(N) < 0.2).astype(float)


def test_group_labels():
    g = group_labels([1, 0, np.nan], [1, 0, 1])
    assert list(g) == ["Emergency - died", "Scheduled - survived", "Scheduled - died"]
    assert set(group_labels(emop, died)) <= set(GROUP_ORDER)


def test_metrics_by_group_handles_single_class_groups():
    p = rng.random(N) * 0.5 + died * 0.4
    groups = np.where(emop > 0.5, "Emergency", "Scheduled")
    t = metrics_by_group(died, p, groups, n_boot=20, threshold=0.5)
    assert "All" in t.index and t.loc["All", "deaths"] == died.sum()
    t2 = metrics_by_group(np.zeros(10), rng.random(10), np.array(["a"] * 10), n_boot=5)
    assert np.isnan(t2.loc["a", "auroc"])


def test_enrichment_detects_planted_signal():
    X = rng.normal(size=(N, 8)); X[:, 0] += 4 * died
    assert knn_death_enrichment(X, died, 10)["enrichment_ratio"] > 3
    Xr = rng.normal(size=(N, 8))
    assert knn_death_enrichment(Xr, died, 10)["enrichment_ratio"] < 2


def test_projection_and_clusters():
    X = rng.normal(size=(N, 20)); X[:, :2] += 5 * died[:, None]
    for m in ("pca", "tsne"):
        assert project_2d(X, m).shape == (N, 2)
    labels, tab = cluster_embeddings(X, k_range=(2, 3))
    assert len(labels) == N and len(tab)
    prof = cluster_profile(labels, died, emop, {"department": rng.choice(["GS", "OS"], N), "asa": rng.integers(1, 5, N)})
    assert prof["n_patients"].sum() == N


def test_cosine_and_plots():
    C = cosine_similarity_matrix(rng.normal(size=(5, 4)))
    assert np.allclose(np.diag(C), 1)
    fig, ax = plt.subplots(1, 2)
    plot_embedding_map(ax[0], rng.normal(size=(N, 2)), group_labels(emop, died))
    names = beeswarm(ax[1], rng.normal(size=(N, 12)), rng.normal(size=(N, 12)), [f"f{i}" for i in range(12)], top=5)
    assert len(names) == 5
    plt.close(fig)
