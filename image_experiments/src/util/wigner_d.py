import functools

import torch
from e3nn.util import explicit_default_types

### https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py

def su2_generators(j, device) -> torch.Tensor:
    m = torch.arange(-j, j).to(device)
    raising = torch.diag(-torch.sqrt(j * (j + 1) - m * (m + 1)), diagonal=-1)

    m = torch.arange(-j + 1, j + 1).to(device)
    lowering = torch.diag(torch.sqrt(j * (j + 1) - m * (m - 1)), diagonal=1)

    m = torch.arange(-j, j + 1).to(device)
    return torch.stack([
        0.5 * (raising + lowering),  # x (usually)
        torch.diag(1j * m),  # z (usually)
        -0.5j * (raising - lowering),  # -y (usually)
    ], dim=0)


def change_basis_real_to_complex(l: int, dtype=None, device=None) -> torch.Tensor:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = torch.zeros((2 * l + 1, 2 * l + 1), dtype=torch.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / 2**0.5
        q[l + m, l - abs(m)] = -1j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1)**m / 2**0.5
        q[l + m, l - abs(m)] = 1j * (-1)**m / 2**0.5
    q = (-1j)**l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real

    dtype, device = explicit_default_types(dtype, device)
    dtype = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }[dtype]
    # make sure we always get:
    # 1. a copy so mutation doesn't ruin the stored tensors
    # 2. a contiguous tensor, regardless of what transpositions happened above
    return q.to(dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format)


def so3_generators(l, device=None) -> torch.Tensor:
    X = su2_generators(l, device)
    Q = change_basis_real_to_complex(l, device=device)
    X = torch.conj(Q.T) @ X @ Q
    assert torch.all(torch.abs(torch.imag(X)) < 1e-5)
    return torch.real(X)


def wigner_D(l, alpha, beta, gamma, X):
    r"""Wigner D matrix representation of :math:`SO(3)`.
    It satisfies the following properties:
    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
    * :math:`D(\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`
    Parameters
    ----------
    l : int
        :math:`l`
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Y axis, applied third.
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around X axis, applied second.
    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Y axis, applied first.
    Returns
    -------
    `torch.Tensor`
        tensor :math:`D^l(\alpha, \beta, \gamma)` of shape :math:`(2l+1, 2l+1)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    alpha = alpha[..., None, None] % (2 * torch.pi)
    beta = beta[..., None, None] % (2 * torch.pi)
    gamma = gamma[..., None, None] % (2 * torch.pi)
    return torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(beta * X[0]) @ torch.matrix_exp(gamma * X[1])