import os
import torch
from torch import Tensor
import torch.nn.functional as F
from src.util.quaternion import quaternion_to_matrix, matrix_to_quaternion, fast_matrix_to_euler_angles_zyz, quaternion_to_matrix_scipy
from einops import einsum, repeat

# ### from https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/irr_repr.py
# DATA_PATH = path = os.path.join(os.path.dirname(__file__), 'data/J_dense.pt')
# Jd = torch.load(DATA_PATH)


def get_regular_rotation(theta):
    s = torch.sin(theta)
    c = torch.cos(theta)
    rot = torch.stack([torch.stack([c, -s], dim=-1),
                       torch.stack([s, c], dim=-1)], dim=-2)
    return rot


class Base_Action():
    action_dim = 0

    def __init__(self, freq, name=None):
        self.freq = freq
        self.name = name

    def rep(self, param: Tensor) -> Tensor:
        '''
        return action_dim x action_dim representation matrix
        '''
        pass

    def trans(self, z, param):
        rep = self.rep(param).to(dtype=z.dtype)
        return einsum(z, rep, 'b h a_in, b a_in a_out -> b h a_out')

    # def z_prehook(self, z):
    #     return z

    # def z_posthook(self, z):
    #     return z


class SO2(Base_Action):
    def __init__(self, freq, name):
        super().__init__(freq, name)
        self.action_dim = 2 if freq > 0 else 1

    def rep(self, param):
        '''
        return 2d rotation matrix
        '''
        if self.freq == 0:
            return param.new_ones(param.shape[0], 1, 1)
        return get_regular_rotation(self.freq * param)

    def invert_param(self, param):
        return -param
    
    def identify(self, z1, z2):
        '''
        estimate the angle theta that minimizes the l2 distance
        ||z2 - rot(theta) z1||^2_Fro
        '''
        if self.freq == 0:
            return None
        zz = einsum(z2, z1, 'b h a1, b h a2 -> b a1 a2')
        theta = torch.atan2(zz[..., 0, 1] - zz[..., 1, 0], 
                            zz[..., 0, 0] + zz[..., 1, 1])
        return theta

    def param_loss(self, param1, param2):
        return 1 - torch.cos(param1 - param2).mean()


class SE2_Rototrans(SO2):
    def __init__(self, freq, name):
        super().__init__(freq, name)
        self.action_dim = 2

    # def rep(self, param):
    #     '''
    #     Return roto-trans matrix.
    #     Param should be a 3-dim vector like: [angle, tx, ty, flag]
    #     if flag == 1, translation is applied *after* rotation (v_new = T R v)
    #     if flag == -1, translation is applied *before* rotation (v_new = R T v)
    #     BE AWARE that M is transposed (i.e. M[batch, row, column])
    #     '''
    #     if self.freq == 0:
    #         return param.new_ones(param.shape[0], 1, 1)
    #     M = torch.zeros(param.shape[0], 3, 3).to(param.device)
    #     R = get_regular_rotation(self.freq * param[:, 0])
    #     t = param[:, 1:3]
    #     M[..., :2, :2] = R
    #     M[..., 2, 0] = t[:, 0]
    #     M[..., 2, 1] = t[:, 1]
    #     M[..., 2, 2] = 1

    #     # Rt = einsum(R, t, 'b a_out a_in, b a_in -> b a_out')
    #     Rt = einsum(t, R, 'b a_in, b a_in a_out -> b a_out')
    #     where_inv = param[:, 3] == -1
    #     M[where_inv, 2, 0] = Rt[where_inv, 0]
    #     M[where_inv, 2, 1] = Rt[where_inv, 1]
    #     return M

    def rep(self, param):
        return super().rep(param[:, 0])

    def trans(self, _z, param):
        z = _z.clone()
        where_inv = param[:, -1] == -1
        ldim = z.shape[1]
        #print(z[where_inv, :].shape, param[where_inv, 1:3].shape)
        z[where_inv] += repeat(param[where_inv, 1:3], 'b a -> b ldim a', ldim=ldim)
        if self.freq != 0:
            z = super().trans(z, param)
        z[~where_inv] += repeat(param[~where_inv, 1:3], 'b a -> b ldim a', ldim=ldim)
        return z

    def invert_param(self, param):
        return -param
    
    def identify(self, z1, z2):
        '''
        estimate the angle theta and translation x, y that minimizes the l2 distance
        ||z2 - M(theta, x, y) z1||^2_Fro
        = ||z2 - T(x, y) R(theta) z1||^2_Fro
        '''
        if self.freq == 0:
            return None
        # zz = einsum(z2, z1, 'b h a1, b h a2 -> b a1 a2')
        # theta = torch.atan2(zz[..., 0, 1] - zz[..., 1, 0], 
        #                     zz[..., 0, 0] + zz[..., 1, 1])
        bsize, ldim = z1.shape[:-1]
        filler = torch.ones(bsize, ldim, 1, device=z1.device)
        z1 = torch.cat([z1, filler], dim=-1)
        z2 = torch.cat([z2, filler], dim=-1)

        param = torch.ones(z1.shape[0], 4, device=z1.device)
        with torch.no_grad():
            M = torch.linalg.lstsq(z1, z2).solution
            param[:, 0] = torch.atan2(M[:, 0, 1], M[:, 0, 0])
            param[:, 1] = M[:, 0, 2]
            param[:, 2] = M[:, 1, 2]
        return param

    # def z_prehook(self, z):
    #     if self.freq == 0:
    #         return z
    #     bsize, ldim = z.shape[:-1]
    #     filler = torch.ones(bsize, ldim, 1)
    #     return torch.cat([z, filler], dim=-1)
    
    # def z_posthook(self, z):
    #     if self.freq == 0:
    #         return z
    #     return z[..., :-1]
    
    # def param_loss(self, param1, param2):
    #     return 1 - torch.cos(param1 - param2).mean()


class SE2_Induced(SE2_Rototrans):
    def __init__(self, freq, name):
        super().__init__(freq, name)
        if freq == 0:
            self.action_dim = 3

    def trans(self, _z, param):
        '''
        For frequency 0, we use first two dims of z for induced representation, which means we apply (tr)^-1 to z[..., :2] where t is translation and r is rotation. 
        For frequence >0, we use the transformation of SO(2).
        '''
        z = _z.clone()
        if self.freq == 0:
            z_induced = z.narrow(dim=-1, start=0, length=2)
            # trans_param = param
            trans_param = -param
            # trans_param[..., 3] *= -1
            z_induced[:] = super().trans(z_induced[:], trans_param)
        else:
            z = SO2.trans(self, z, param)
        return z
        


class SO3(Base_Action):
    action_dim = 3
    invert_scale = torch.tensor([1, -1, -1, -1])

    def rep(self, param):
        '''
        return 3d rotation matrix
        '''
        return quaternion_to_matrix(param)

    def invert_param(self, param):
        if self.invert_scale.device != param.device:
            self.invert_scale = self.invert_scale.to(param.device)
        #return quaternion_invert(param)
        return self.invert_scale * param

    def identify_as_rot(self, z1, z2):
        '''
        estimate the angle theta that minimizes the l2 distance
        ||z2 - rot(theta) z1||^2_Fro.
        Using SVD: https://en.wikipedia.org/wiki/Wahba%27s_problem#Solution_via_SVD
        '''

        zz = einsum(z2, z1, 'b h a2, b h a1 -> b a2 a1')
        if z1.dtype == torch.float16:
            zz = zz.to(dtype=torch.float32)
        U, _, Vt = torch.linalg.svd(zz)
        vol = torch.linalg.det(U) * torch.linalg.det(Vt)
        if z1.dtype == torch.float16:
            U, Vt, vol = U.half(), Vt.half(), vol.half()
        m = torch.stack([torch.ones_like(vol), torch.ones_like(vol), vol], dim=-1)
        rot = einsum(U, m, Vt, 'b i j, b j, b j k -> b k i')
        return rot

    def identify(self, z1, z2):
        rot = self.identify_as_rot(z1, z2)
        return matrix_to_quaternion(rot)

    def param_loss(self, param1, param2):
        return (1 - einsum(param1, param2, 'b p, b p -> b').abs()).mean()

# class SO3_WignerD(SO3):
#     def __init__(self, freq, name=None):
#         super().__init__(freq, name)
#         self.action_dim = 2 * freq + 1
#         self.X = so3_generators(freq).detach()

#     def rep(self, param):
#         '''
#         return Wigner-D matrix
#         '''
#         if self.X.device != param.device:
#             self.X = self.X.to(param.device)

#         # with torch.no_grad():
#         #     rot = quaternion_to_matrix(param)
#         #     alpha, beta, gamma = matrix_to_euler_angles(rot, 'YXY').unbind(dim=-1)
#         #     return wigner_D(self.freq, alpha, beta, gamma, self.X)

#         with torch.no_grad():
#             alpha, beta, gamma = quaternion_to_euler_angles(param, 'YXY').unbind(dim=-1)
#             return wigner_D(self.freq, alpha, beta, gamma, self.X)

#     def rep_from_euler(self, alpha, beta, gamma):
#         with torch.no_grad():
#            return wigner_D(self.freq, alpha, beta, gamma, self.X)


def to_order(degree):
    return 2 * degree + 1

def quaternion_to_euler_angle(quat):
    quat = F.normalize(quat, dim=-1)
    rot = quaternion_to_matrix_scipy(quat)
    # alpha, beta, gamma = matrix_to_euler_angles_zyz(rot).unbind(dim=-1)
    alpha, beta, gamma = fast_matrix_to_euler_angles_zyz(rot)
    return [alpha, beta, gamma]

class SO3_WignerD_ZYZ(SO3):
    def __init__(self, freq, data_path, name=None):
        degree = freq
        super().__init__(degree, name)
        order = to_order(degree)
        self.action_dim = order

        self.J = torch.load(data_path)[degree]
        self.inds = torch.arange(0, order, 1, dtype=torch.long)
        self.reversed_inds = torch.arange(2 * degree, -1, -1, dtype=torch.long)
        self.frequencies = torch.arange(degree, -degree - 1, -1)
        
        self.order = order
        self.degree = degree
        
    def wigner_d_matrix(self, alpha, beta, gamma):
        """Create wigner D matrices for batch of ZYZ Euler anglers for degree l."""
        if self.inds.device != alpha.device or self.J.dtype != alpha.dtype:
            self.cast_attrs(alpha)
        x_a = self.z_rot_mat(alpha)
        x_b = self.z_rot_mat(beta)
        x_c = self.z_rot_mat(gamma)
        res = x_a @ self.J @ x_b @ self.J @ x_c
        return res

    def z_rot_mat(self, angle):
        m = angle.new_zeros(angle.shape[0], self.order, self.order)

        angle_freq = torch.outer(angle, self.frequencies)
        m[..., self.inds, self.reversed_inds] = torch.sin(angle_freq)
        m[..., self.inds, self.inds] = torch.cos(angle_freq)
        return m
        
    def rep(self, quat):
        '''
        return Wigner-D matrix
        '''
        if self.freq == 0:
            return quat.new_ones(quat.shape[0], 1, 1)
        
        with torch.no_grad():
            alpha, beta, gamma = quaternion_to_euler_angle(quat)
            return self.wigner_d_matrix(alpha, beta, gamma)

    # def _rep(self, euler_angles):
    #     '''
    #     return Wigner-D matrix
    #     '''
    #     alpha, beta, gamma = euler_angles
    #     if self.freq == 0:
    #         return alpha.new_ones(len(alpha), 1, 1)
    #     return self.wigner_d_matrix(alpha, beta, gamma)

    def cast_attrs(self, tensor):
        dtype, device = tensor.dtype, tensor.device
        self.J = self.J.type(dtype).to(device)
        self.inds = self.inds.to(device)
        self.reversed_inds = self.reversed_inds.to(device)
        self.frequencies = self.frequencies.type(dtype).to(device)

    def identify_as_rot(self, z1, z2):
        '''
        change coordinate from zyz to yxy
        '''
        rot = super().identify_as_rot(z1, z2)
        permutation = [2, 0, 1]
        rot[:] = rot[..., :, permutation]
        rot[:] = rot[..., permutation, :]
        return rot



class SO3_And_Zoom(SO3_WignerD_ZYZ):
    def __init__(self, freq, data_path, name=None, inv_scale=False):
        super().__init__(freq, data_path, name)
        self.inv_scale_param = inv_scale

    def decompose_param(self, param):
        quat = param[:, :4]
        _scale = param[:, 4:]
        scale = 1 / _scale if self.inv_scale_param else _scale            
        return quat, scale

    def rep(self, param):
        '''
        return spherical harmonics multipied by scale that represents zoom in/out (focal length)
        '''
        quat, scale = self.decompose_param(param)
        return super().rep(quat) / scale.unsqueeze(dim=-1)

    def invert_param(self, param):
        quat, scale = self.decompose_param(param)
        if self.invert_scale.device != quat.device:
            self.invert_scale = self.invert_scale.to(quat.device)
        return torch.cat([self.invert_scale * quat, 1 / scale], dim=-1)

    def param_loss(self, param1, param2):
        quat1, _ = self.decompose_param(param1)
        quat2, _ = self.decompose_param(param2)
        return super().param_loss(quat1, quat2)



if __name__ == '__main__':
    quat = torch.Tensor([[ 0.1241,  0.2008,  0.9328, -0.2722],
                        [ 0.4370, -0.6642, -0.3006, -0.5268],
                        [ 0.6012, -0.2805, -0.5556, -0.5012]])
    so3 = SO3_WignerD(freq=1, name='test')
    print(quaternion_to_matrix(quat))
    print(so3.rep(quat))