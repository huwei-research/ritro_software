"""Compatibility support for the manuscript's adapted GaBO comparison.

The upstream SPD kernel is included with its MIT notice under vendor/gabo.
Only the kernel is used; acquisition fitting and proposal code are local ports.
"""
import numpy as np
import torch
import gpytorch
from vendor.gabo import kernels_spd as author
from vendor.gabo.spd_utils_torch import symmetric_matrix_to_vector_mandel_torch as mandel
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from ritro.manifolds import SPD

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

def distance(A,B,diagonal_distance=False):
    # Cholesky whitening has the same generalized eigenvalues as symmetric
    # whitening, but uses supported, vectorized modern PyTorch operations.
    if diagonal_distance:
        C=torch.linalg.solve_triangular(torch.linalg.cholesky(A),B,upper=False)
        C=torch.linalg.solve_triangular(torch.linalg.cholesky(A),C.transpose(-1,-2),upper=False).transpose(-1,-2)
    else:
        L=torch.linalg.cholesky(A).unsqueeze(-3);Y=B.unsqueeze(-4)
        C=torch.linalg.solve_triangular(L,Y,upper=False)
        C=torch.linalg.solve_triangular(L,C.transpose(-1,-2),upper=False).transpose(-1,-2)
    w=torch.linalg.eigvalsh((C+C.transpose(-1,-2))/2)
    return torch.linalg.vector_norm(torch.log(w),dim=-1)

author.affine_invariant_distance_torch=distance
_author_unmandel=author.vector_to_symmetric_matrix_mandel_torch
author.vector_to_symmetric_matrix_mandel_torch=lambda x: _author_unmandel(x.contiguous())
BASIS=torch.tensor(SPD().basis)
LOW,HIGH=20.,5000.

def matrix(z):return torch.matrix_exp(torch.einsum('...i,ijk->...jk',z,BASIS))

def logs(X):
    w,Q=np.linalg.eigh(X)
    return np.einsum('ijk,jk->i',SPD().basis,(Q*np.log(w))@Q.T)

def sample(rng,n):
    result=[]
    for _ in range(n):
        Q=np.linalg.qr(rng.normal(size=(3,3)))[0];w=rng.uniform(LOW,HIGH,3)
        result.append(logs((Q*w)@Q.T))
    return np.array(result)
