import numpy as np
import torch
from scipy.spatial.transform import Rotation
import scipy.linalg
torch_pi = torch.tensor(3.1415926535)

# hat map from vector space R^3 to Lie algebra so(3)
def hat(v):
    """
    v: [..., 3]
    hat_v: [..., 3, 3]
    """
    hat_v = torch.zeros([*v.shape[:-1], 3, 3], dtype=v.dtype, device=v.device)
    hat_v[..., 0, 1], hat_v[..., 0, 2], hat_v[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    return hat_v + -hat_v.transpose(-1, -2)

# vee map from Lie algebra so(3) to the vector space R^3
def vee(A):
    if not torch.allclose(A, -A.transpose(-1, -2), atol=1e-4, rtol=1e-4):
        print("Input A must be skew symmetric, Err" + str(((A - A.transpose(-1,
            -2))**2).sum(dim=[-1, -2])))
    vee_A = torch.stack([-A[..., 1, 2], A[..., 0, 2], -A[..., 0, 1]], dim=-1)
    return vee_A

# Logarithmic map from SO(3) to R^3 (i.e. rotation vector)
def Log(R):
    shape = list(R.shape[:-2])
    R_ = R.reshape([-1, 3, 3]).to(torch.float64)
    Log_R_ = rotation_vector_from_matrix(R_)
    return Log_R_.reshape(shape + [3]).to(R.dtype)

# logarithmic map from SO(3) to so(3), this is the matrix logarithm
def log(R): return hat(Log(R))

# Exponential map from so(3) to SO(3), this is the matrix exponential
def exp(A): return torch.matrix_exp(A)

# Exponential map from R^3 to SO(3)
def Exp(A): return exp(hat(A))

# Angle of rotation SO(3) to R^+, this is the norm in our chosen orthonormal basis
def Omega(R, eps=1e-6):
    # multiplying by (1-epsilon) prevents instability of arccos when provided with -1 or 1 as input.
    R_ = R.to(torch.float64)
    assert not torch.any(torch.abs(R) > 1.1)
    trace = torch.diagonal(R_, dim1=-2, dim2=-1).sum(dim=-1) * (1-eps)
    out = (trace - 1.)/2.
    out = torch.clamp(out, min=-0.9999, max=0.9999)
    return torch.arccos(out).to(R.dtype)

# exponential map from tangent space at R0 to SO(3)
def expmap(R0, tangent):
    skew_sym = torch.einsum('...ij,...ik->...jk', R0, tangent)
    if not  torch.allclose(skew_sym, -skew_sym.transpose(-1, -2), atol=1e-3,
            rtol=1e-3):
        print("in expmap, R0.T @ tangent must be skew symmetric")
    skew_sym = (skew_sym - torch.transpose(skew_sym, -2, -1))/2.
    exp_skew_sym = exp(skew_sym)
    return torch.einsum('...ij,...jk->...ik', R0, exp_skew_sym)

# Logarithmic map from SO(3) to tangent space at R0
def logmap(R0, R):
    R0_transpose_R = torch.einsum('...ij,...ik->...jk', R0, R)
    return torch.einsum(
        '...ij,...jk->...ik',
        R0, log(R0_transpose_R)
    )

    # Normal sample in tangent space at R0
def tangent_gaussian(R0): return torch.einsum('...ij,...jk->...ik', R0,
        hat(torch.randn(*R0.shape[:-2], 3, dtype=R0.dtype, device=R0.device)))

# Usual log density of normal distribution in Euclidean space
def normal_log_density(x, mean, var):
    return (-(1/2)*(x-mean)**2 / var - 
            ((1/2)*torch.log(2*torch_pi*var)).to(x.device).to(x.device)).sum(dim=-1)

# log density of Gaussian in the tangent space
def tangent_gaussian_log_density(R, R_mean, var):
    Log_RmeanT_R = Log(torch.einsum('Nji,Njk->Nik', R_mean, R))
    return normal_log_density(Log_RmeanT_R, torch.zeros_like(Log_RmeanT_R), var)

# sample from uniform distribution on SO(3)
def sample_uniform(N, M=1000):
    omega_grid = np.linspace(0, np.pi, M)
    cdf = np.cumsum(np.pi**-1 * (1-np.cos(omega_grid)), 0)/(M/np.pi)
    omegas = np.interp(np.random.rand(N), cdf, omega_grid)
    axes = np.random.randn(N, 3)
    axes = omegas[..., None]* axes/np.linalg.norm(axes, axis=-1, keepdims=True)
    axes_ = axes.reshape([-1, 3])
    Rs = exp(hat(torch.tensor(axes_)))
    Rs = Rs.reshape([N, 3, 3])
    return Rs



### New Log map adapted from geomstats
def rotation_vector_from_matrix(rot_mat):
    """Convert rotation matrix (in 3D) to rotation vector (axis-angle).

    # Adapted from geomstats
    # https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py#L884

    Get the angle through the trace of the rotation matrix:
    The eigenvalues are:
    :math:`\{1, \cos(angle) + i \sin(angle), \cos(angle) - i \sin(angle)\}`
    so that:
    :math:`trace = 1 + 2 \cos(angle), \{-1 \leq trace \leq 3\}`
    The rotation vector is the vector associated to the skew-symmetric
    matrix
    :math:`S_r = \frac{angle}{(2 * \sin(angle) ) (R - R^T)}`

    For the edge case where the angle is close to pi,
    the rotation vector (up to sign) is derived by using the following
    equality (see the Axis-angle representation on Wikipedia):
    :math:`outer(r, r) = \frac{1}{2} (R + I_3)`
    In nD, the rotation vector stores the :math:`n(n-1)/2` values
    of the skew-symmetric matrix representing the rotation.

    Parameters
    ----------
    rot_mat : array-like, shape=[..., n, n]
        Rotation matrix.

    Returns
    -------
    regularized_rot_vec : array-like, shape=[..., 3]
        Rotation vector.
    """
    angle = Omega(rot_mat)
    assert len(angle.shape)==1, "cannot handle vectorized Log map here"
    n_rot_mats = len(angle)
    rot_mat_transpose = torch.transpose(rot_mat, -2, -1)
    rot_vec_not_pi = vee(rot_mat - rot_mat_transpose)
    mask_0 = torch.isclose(angle, torch.tensor(0.0, dtype=angle.dtype, device=angle.device)).to(angle.dtype)
    mask_pi = torch.isclose(angle, torch_pi.to(angle.dtype), atol=1e-2).to(angle.dtype)
    mask_else = (1 - mask_0) * (1 - mask_pi)

    numerator = 0.5 * mask_0 + angle * mask_else
    denominator = (
        (1 - angle**2 / 6) * mask_0 + 2 * torch.sin(angle) * mask_else + mask_pi
    )

    rot_vec_not_pi = rot_vec_not_pi * numerator[..., None] / denominator[..., None]

    vector_outer = 0.5 * (torch.eye(3, dtype=rot_mat.dtype, device=rot_mat.device) + rot_mat)
    vector_outer = vector_outer + (torch.maximum(torch.tensor(0.0,
        dtype=vector_outer.dtype, device=rot_mat.device), vector_outer) - vector_outer)*torch.eye(3,
                dtype=vector_outer.dtype, device=rot_mat.device)
    squared_diag_comp = torch.diagonal(vector_outer, dim1=-2, dim2=-1)
    diag_comp = torch.sqrt(squared_diag_comp)
    norm_line = torch.linalg.norm(vector_outer, dim=-1)
    max_line_index = torch.argmax(norm_line, dim=-1)
    selected_line = vector_outer[range(n_rot_mats), max_line_index]
    # want
    signs = torch.sign(selected_line)
    rot_vec_pi = angle[..., None] * signs * diag_comp

    rot_vec = rot_vec_not_pi + mask_pi[..., None] * rot_vec_pi
    return regularize(rot_vec)

def regularize(point):
    """Regularize a point to be in accordance with convention.
    In 3D, regularize the norm of the rotation vector,
    to be between 0 and pi, following the axis-angle
    representation's convention.
    If the angle is between pi and 2pi,
    the function computes its complementary in 2pi and
    inverts the direction of the rotation axis.
    Parameters

    # Adapted from geomstats
    # https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py#L884
    ----------
    point : array-like, shape=[...,3]
        Point.
    Returns
    -------
    regularized_point : array-like, shape=[..., 3]
        Regularized point.
    """
    theta = torch.linalg.norm(point, axis=-1)
    k = torch.floor(theta / 2.0 / torch_pi)

    # angle in [0;2pi)
    angle = theta - 2 * k * torch_pi

    # this avoids dividing by 0
    theta_eps = torch.where(torch.isclose(theta, torch.tensor(0.0,
        dtype=theta.dtype)), 1.0, theta)

    # angle in [0, pi]
    normalized_angle = torch.where(angle <= torch_pi, angle, 2 * torch_pi - angle)
    norm_ratio = torch.where(torch.isclose(theta, torch.tensor(0.0,
        dtype=theta.dtype)), 1.0, normalized_angle / theta_eps)

    # reverse sign if angle was greater than pi
    norm_ratio = torch.where(angle > torch_pi, -norm_ratio, norm_ratio)
    return torch.einsum("...,...i->...i", norm_ratio, point)

def taylor_exp_even_func(point, taylor_function, order=5, tol=1e-6):
    """Taylor Approximation of an even function around zero.
    Parameters
    ----------
    point : array-like
        Argument of the function to approximate.
    taylor_function : dict with following keys
        function : callable
            Even function to approximate around zero.
        coefficients : list
            Taylor coefficients of even order at zero.
    order : int
        Order of the Taylor approximation.
        Optional, Default: 5.
    tol : float
        Threshold to use the approximation instead of the function's value.
        Where `abs(point) <= tol`, the approximation is returned.
    Returns
    -------
    function_value: array-like
        Value of the function at point.
    """
    approx = torch.einsum(
        "k,k...->...",
        torch.tensor(taylor_function["coefficients"][:order], device=point.device, dtype=point.dtype),
        torch.stack([point**k for k in range(order)]),
    )
    point_ = torch.where(torch.abs(point) <= tol, tol, point)
    exact = taylor_function["function"](torch.sqrt(point_))
    result = torch.where(torch.abs(point) < tol, approx, exact)
    return result

import math
COS_TAYLOR_COEFFS = [
    1.0,
    -1.0 / math.factorial(2),
    +1.0 / math.factorial(4),
    -1.0 / math.factorial(6),
    +1.0 / math.factorial(8),
]
SINC_TAYLOR_COEFFS = [
    1.0,
    -1.0 / math.factorial(3),
    +1.0 / math.factorial(5),
    -1.0 / math.factorial(7),
    +1.0 / math.factorial(9),
]
cos_close_0 = {"function": torch.cos, "coefficients": COS_TAYLOR_COEFFS}
sinc_close_0 = {"function": lambda x: torch.sin(x) / x, "coefficients": SINC_TAYLOR_COEFFS}

def quaternion_from_rotation_vector(rot_vec):
    """Convert a rotation vector into a unit quaternion.
    Parameters
    ----------
    rot_vec : array-like, shape=[..., 3]
        Rotation vector.
    Returns
    -------
    quaternion : array-like, shape=[..., 4]
        Quaternion.
    """
    init_dtype = rot_vec.dtype

    rot_vec = rot_vec.to(torch.float64)

    rot_vec = regularize(rot_vec)

    squared_angle = torch.sum(rot_vec**2, axis=-1)

    coef_cos = taylor_exp_even_func(squared_angle / 4, cos_close_0)
    coef_sinc = 0.5 * taylor_exp_even_func(
        squared_angle / 4, sinc_close_0
    )

    quaternion = torch.concatenate(
        (coef_cos[..., None], torch.einsum("...,...i->...i", coef_sinc, rot_vec)),
        axis=-1,
    )

    return quaternion.to(init_dtype)

def quaternion_from_matrix(rot_mat):
    """Convert a rotation matrix into a unit quaternion.
    Parameters
    ----------
    rot_mat : array-like, shape=[..., 3, 3]
        Rotation matrix.
    Returns
    -------
    quaternion : array-like, shape=[..., 4]
        Quaternion.  First dim is the real part (cos(theta/2)
    """
    rot_vec = Log(rot_mat)
    return quaternion_from_rotation_vector(rot_vec)
