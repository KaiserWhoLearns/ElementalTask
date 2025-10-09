# tests/test_fv_basis.py
import numpy as np
import pytest

# Adjust this import to match your module filename (e.g., extract_function_vecs.py)
from function_vecs.extract_function_vecs import (
    Headset, TaskHeadMeans, TaskFunctionVec, TaskMatrix, SkillBasis,
    build_function_vec_from_means, stack_function_vecs, build_skill_basis
)

# -------------------------
# Fixtures / helpers
# -------------------------
@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(0)
    yield

def l2norm(x):
    return np.linalg.norm(x)

# -------------------------
# Tests for FV construction
# -------------------------

def test_build_function_vec_topk_sums_columns_and_normalizes():
    d, H = 8, 5
    means = np.random.randn(d, H).astype(np.float32)
    head_means = TaskHeadMeans(task_name="toy_task", residual_means=means)
    headset = Headset(mode="topk", heads=[(0,i) for i in range(H)], weights=None)

    fv = build_function_vec_from_means(head_means, headset, normalization="l2")
    # Expected: column-sum
    expected = means.sum(axis=1)
    expected = expected / (np.linalg.norm(expected) + 1e-12)

    assert isinstance(fv, TaskFunctionVec)
    assert fv.function_vec.shape == (d,)
    assert np.allclose(fv.function_vec, expected, atol=1e-6)
    assert pytest.approx(1.0, rel=1e-6) == l2norm(fv.function_vec)

def test_build_function_vec_soft_weighted_sum():
    d, H = 8, 4
    means = np.random.randn(d, H).astype(np.float32)
    w = np.array([0.1, 0.3, 0.4, 0.2], dtype=np.float32)
    head_means = TaskHeadMeans(task_name="toy_task", residual_means=means)
    headset = Headset(mode="soft", heads=[(0,i) for i in range(H)], weights=w)

    fv = build_function_vec_from_means(head_means, headset, normalization="none")
    expected = means @ w

    assert np.allclose(fv.function_vec, expected, atol=1e-6)
    assert fv.normalization == "none"

def test_stack_function_vecs_column_stack_shape_and_names():
    d = 6
    f1 = TaskFunctionVec(task_name="t1", function_vec=np.random.randn(d).astype(np.float32))
    f2 = TaskFunctionVec(task_name="t2", function_vec=np.random.randn(d).astype(np.float32))
    tm = stack_function_vecs([f1, f2])

    assert isinstance(tm, TaskMatrix)
    assert tm.V.shape == (d, 2)          # column-stacked
    assert tm.task_names == ["t1", "t2"]

# -------------------------
# SVD / basis tests
# -------------------------

def test_build_skill_basis_energy_auto_k_and_reconstruction():
    """
    Construct a rank-3, zero-mean matrix V = U_true diag(S_true) R^T with chosen spectrum.
    With S_true^2 = [25, 4, 1], 95% energy requires k=2 (since 25/30=0.833, (25+4)/30=0.967).
    """
    d, m, r_true = 20, 50, 3

    # Orthonormal U_true (d x r_true) and R (m x r_true)
    U_random, _ = np.linalg.qr(np.random.randn(d, r_true))
    R_random, _ = np.linalg.qr(np.random.randn(m, r_true))

    S_true = np.array([5.0, 2.0, 1.0], dtype=np.float64)   # squared: 25,4,1 -> total 30

    # V_centered = U diag(S_true) R^T ; mean is exactly zero
    Vc = (U_random @ np.diag(S_true) @ R_random.T).astype(np.float64)

    # Build basis (your function centers internally; here mean is already ~0)
    tm = TaskMatrix(V=Vc.astype(np.float32), task_names=[f"t{i}" for i in range(m)])
    basis = build_skill_basis(tm, method="svd", k=-1)

    # Check k chosen by energy (should be 2 for 95%)
    k_expected = 2
    assert basis.U.shape == (d, k_expected)
    assert basis.S.shape == (k_expected,)
    assert basis.Vt.shape == (k_expected, m)

    # Reconstruct and check relative error small (because we kept 95%+ energy)
    U_k, S_k, Vt_k = basis.U.astype(np.float64), basis.S.astype(np.float64), basis.Vt.astype(np.float64)
    V_hat = (U_k @ np.diag(S_k) @ Vt_k)
    rel_err = np.linalg.norm(Vc - V_hat, ord="fro") / (np.linalg.norm(Vc, ord="fro") + 1e-12)
    assert rel_err < 0.25   # with 95% energy kept, residual energy ≈ 5%; Frobenius error <~ sqrt(0.05) ~ 0.224

def test_build_skill_basis_shapes_and_orthogonality():
    d, m = 16, 10
    V = np.random.randn(d, m).astype(np.float32)
    tm = TaskMatrix(V=V, task_names=[f"t{i}" for i in range(m)])
    basis = build_skill_basis(tm, method="svd", k=-1)

    U = basis.U.astype(np.float64)
    I = U.T @ U
    assert I.shape == (U.shape[1], U.shape[1])
    # U columns should be (approximately) orthonormal
    assert np.allclose(I, np.eye(I.shape[0]), atol=1e-5)
