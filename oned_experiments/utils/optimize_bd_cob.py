import torch
import torch.nn as nn
from einops import repeat
from utils.laplacian import tracenorm_of_normalized_laplacian, make_identity_like
from tqdm import tqdm
from utils.gobtainer import ChangeOfBasis
import numpy as np


'''
Automated Block Diagonalization (Maehara et al.)
This code is the python rendition of the algorithm publicized at
http://www.misojiro.t.u-tokyo.ac.jp/~maehara/commdec/

'''

from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix

def comm(A, X):
    return A @ X - X @ A

def multiply(A, v):
    n = A[0].shape[1]
    N = len(A)
    X = np.reshape(v, (n, n))
    W = np.zeros((n, n))
    for k in range(N):
        W += comm(A[k].T, comm(A[k], X)) + comm(A[k], comm(A[k].T, X))
    W += np.eye(n) * np.trace(X)
    return W.reshape(n*n)


'''
Simultaneously Block Diagonalizing N matrices of shape n x n.
A :the matrix of shape  N x n x n. 
'''
def commdec(A, printSW=1, pickup=10, krylovth=1e-8):
    N = len(A)
    n = A[0].shape[1]

    # Settings
    krylovth = krylovth
    krylovit = n**2
    maxdim = n
    pickup = pickup

    v = [np.random.randn(n**2)]
    v[0] /= np.sqrt(v[0] @ v[0])
    H = csc_matrix((n**2, n**2), dtype=float)

    for j in range(krylovit):
        w = multiply(A, v[j])
        for i in range(max(0, j-1), j+1):
            H[i, j] = v[i] @ w
            w -= H[i, j] * v[i]
        a = np.sqrt(w @ w)

        if (a < krylovth) or (j == krylovit-1):
            break

        H[j+1, j] = a
        v.append(w / a)

    H = H[:j+1, :j+1]
    H = (H + H.T) / 2
    Q = np.column_stack(v)

    d, Y= eigs(H, k=maxdim, which='SM')
    #print(Y.shape, d.shape)

    Y = Y.reshape([Y.shape[0], -1])
    d = np.sqrt(np.diag(d)/(4*n))
    e = d[pickup-1]

    X = np.zeros([n**2,1]).astype(np.complex128)

    for i in range(pickup):
        #print(X.shape, Y[:, [0]].shape)
        X = X + Y[:,[i]]
    X = np.reshape(Q @ X, (n, n))
    D, P = np.linalg.eig(X + X.T)

    if printSW > 0:
        print(f"""err = {e}""")
    if printSW > 1:
        print('small eigs (normalized):')
        len_d = min(3+pickup, d.shape[0])
        for i in range(len_d):
            print(f'{i+1}: err = {d[i]:.3e}')



    return P, e


######################################################################
def optimize_bd_cob(mats, batchsize=32, n_epochs=50, epochs_monitor=10, lr=0.1,
num_workers=0, device='cpu', orthloss=0, modelnow=None):
    # Optimize change of basis matrix U by minimizing block diagonalization loss



    if modelnow is not None:
        change_of_basis = modelnow.to(device)
    else:
        change_of_basis = ChangeOfBasis(mats.shape[-1]).to(device)
    dataloader = torch.utils.data.DataLoader(
        mats, batch_size=batchsize, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(change_of_basis.parameters(), lr=lr)
    for ep in range(n_epochs):
        total_loss, total_N = 0, 0
        for mat in dataloader:
            mat = mat.to(device)
            n_mat = change_of_basis(mat)
            n_mat = torch.abs(n_mat)
            n_mat = torch.matmul(n_mat.transpose(-2, -1), n_mat)
            loss = torch.mean(
                tracenorm_of_normalized_laplacian(n_mat))
            if orthloss >0:
                loss = loss + change_of_basis.orthloss()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mat.shape[0]
            total_N += mat.shape[0]
        if ((ep+1) % epochs_monitor) == 0:
            print('ep:{} loss:{}'.format(ep, total_loss/total_N))
    return change_of_basis
