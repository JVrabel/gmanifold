"""Shared neighbour searches and batched (vmapped) fits; runs on CPU or CUDA."""
import torch
import pytest
import gmanifold as gm
from gmanifold.geometry import knn

dev = "cuda" if torch.cuda.is_available() else "cpu"


def sheet(n, seed, D=32):
    """n points of one fixed curved 2-D sheet in R^D (the sheet is the same for every seed; the points differ)."""
    A = torch.linalg.qr(torch.randn(D, 3, device=dev, generator=torch.Generator(device=dev).manual_seed(0)))[0].T * 3
    st = torch.rand(n, 2, device=dev, generator=torch.Generator(device=dev).manual_seed(seed))
    return torch.stack([st[:, 0], st[:, 1], 0.8 * torch.sin(3 * st[:, 0])], 1) @ A


def test_precomputed_neighbours_match():
    X = sheet(1500, 0)
    X[:3] = X[3:6] + 1e-4                                     # near-duplicates
    X[6] = X[6] + 50                                          # an outlier
    nn = knn(X, 32)
    assert torch.equal(gm.degenerate_mask(X, nn=nn), gm.degenerate_mask(X))
    assert torch.equal(gm.outlier_mask(X, nn=nn), gm.outlier_mask(X))
    keep = ~gm.degenerate_mask(X) & ~gm.outlier_mask(X); Xk = X[keep]
    nnk = knn(Xk, 40)                                         # K' > K is fine: the first K columns are used
    a, b = gm.intrinsic_dimension(Xk, nn=nnk), gm.intrinsic_dimension(Xk)
    assert all(abs(a[k] - b[k]) < 1e-5 for k in a)
    with pytest.raises(ValueError):
        gm.intrinsic_dimension(Xk, K=32, nn=knn(Xk, 8))


def test_fit_with_precomputed_neighbours_is_identical():
    X = sheet(1200, 1)
    kw = dict(epochs=5, X_val=X[:100], seed=0, log_every=2)
    A = gm.GlobalManifold(latent_dim=2, hidden=(32, 16), device=dev).fit(X, **kw)
    B = gm.GlobalManifold(latent_dim=2, hidden=(32, 16), device=dev).fit(X, nn=knn(X, 48), **kw)
    for ra, rb in zip(A.history, B.history):                  # equal up to the summation order of tied neighbours
        assert ra.keys() == rb.keys() and all(abs(ra[k] - rb[k]) <= 1e-5 * (1 + abs(ra[k])) for k in ra)
    assert torch.equal(A.spacing, B.spacing) and torch.allclose(A.encode(X), B.encode(X), atol=1e-4, rtol=1e-3)


def test_fit_many_matches_fit_for_one_cloud():
    """Without the curvature term (its random directions are drawn differently under vmap) a batch of one cloud
    follows exactly the same optimisation as `fit`, up to batched-matmul rounding."""
    X = sheet(1024, 2); Xv = sheet(100, 3)                    # N divisible by the batch size: no wrap-around
    kw = dict(epochs=12, batch_size=128, lam_curv=0.0, seed=0, log_every=4)
    A = gm.GlobalManifold(latent_dim=2, hidden=(32, 16), device=dev).fit(X, X_val=Xv, **kw)
    (B,) = gm.fit_many([gm.GlobalManifold(latent_dim=2, hidden=(32, 16), device=dev)], [X], X_vals=[Xv], **kw)
    for ra, rb in zip(A.history, B.history):
        assert ra.keys() == rb.keys()
        for k in ra:
            assert abs(ra[k] - rb[k]) <= 1e-4 * (1 + abs(ra[k])), (k, ra, rb)
    assert torch.allclose(A.encode(Xv), B.encode(Xv), atol=1e-4, rtol=1e-3)
    assert torch.allclose(A.rho, B.rho, atol=1e-4, rtol=1e-3)


def test_fit_many_clouds_of_different_sizes():
    Xs = [sheet(n, 10 + i) for i, n in enumerate((1500, 900, 1200))]
    Xvs = [sheet(150, 20 + i) for i in range(3)]
    nn = [knn(X, 32) for X in Xs]
    models = [gm.GlobalManifold(latent_dim=2, hidden=(64, 32), device=dev) for _ in Xs]
    gm.fit_many(models, Xs, epochs=60, batch_size=256, X_vals=Xvs, nn=nn, seed=0, log_every=20)
    for M, X, Xv in zip(models, Xs, Xvs):
        assert len(M.history) == 60 and "val_recon_over_spacing" in M.history[-1]
        assert M.history[-1]["val_recon_over_spacing"] < 0.5
        assert M.Z.shape == (len(X), 2) and M.rho.shape == (len(X),)
        assert not M.training
    # the models are independent: refitting one of them alone gives the same kind of result, and the others differ
    d01 = (models[0].encode(Xvs[0]) - models[1].encode(Xvs[0])).abs().max()
    assert d01 > 1e-3


def test_fit_many_rejects_mismatched_models():
    X = sheet(300, 5)
    with pytest.raises(ValueError):
        gm.fit_many([gm.GlobalManifold(2, hidden=(16, 8), device=dev), gm.GlobalManifold(3, hidden=(16, 8), device=dev)], [X, X], epochs=1)


def test_tangent_charts_with_precomputed_neighbours():
    X = sheet(800, 7); Y = sheet(50, 8)
    A = gm.TangentCharts(X, 2); B = gm.TangentCharts(X, 2, nn=knn(X, 40))
    assert torch.equal(A.idx, B.idx) and torch.allclose(A.residual(Y), B.residual(Y))
    with pytest.raises(ValueError):
        gm.TangentCharts(X, 2, nn=knn(X, 8))
