# subsample layer for 3d processing.
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Function
import math
import numpy as np
# from openpoints.cpp.pointnet2_batch import pointnet2_cuda


class BaseSampler(ABC):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def __init__(self, ratio=None, num_to_sample=None, subsampling_param=None):
        if num_to_sample is not None:
            if (ratio is not None) or (subsampling_param is not None):
                raise ValueError(
                    "Can only specify ratio or num_to_sample or subsampling_param, not several !")
            self._num_to_sample = num_to_sample

        elif ratio is not None:
            self._ratio = ratio

        elif subsampling_param is not None:
            self._subsampling_param = subsampling_param

        else:
            raise Exception(
                'At least ["ratio, num_to_sample, subsampling_param"] should be defined')

    def __call__(self, xyz):
        return self.sample(xyz)

    def _get_num_to_sample(self, npoints) -> int:
        if hasattr(self, "_num_to_sample"):
            return self._num_to_sample
        else:
            return math.floor(npoints * self._ratio)

    def _get_ratio_to_sample(self, batch_size) -> float:
        if hasattr(self, "_ratio"):
            return self._ratio
        else:
            return self._num_to_sample / float(batch_size)

    @abstractmethod
    def sample(self, xyz, feature=None, batch=None):
        pass


class RandomSample(BaseSampler):
    """Random Sample for dense data
        Arguments:
            xyz -- [B, N, 3]
    """

    def sample(self, xyz, **kwargs):
        if len(xyz.shape) != 3:
            raise ValueError(" Expects the xyz tensor to be of dimension 3")
        B, N, _ = xyz.shape
        idx = torch.randint(
            0, N, (B, self._get_num_to_sample(N)), device=xyz.device)
        sampled_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        # sampled_feature = torch.gather(feature, 2, idx.unsqueeze(1).repeat(1, C, 1))
        return sampled_xyz, idx


def random_sample(xyz, npoint):
    B, N, _ = xyz.shape
    idx = torch.randint(0, N, (B, npoint), device=xyz.device)
    return idx


def furthest_point_sampling_wrapper(b, n, m, points_tensor, temp_tensor, idx_tensor):
    """
    采用最远点采样（Furthest Point Sampling, FPS）来选择 `m` 个点，使得它们的最小距离最大化。

    :param b: 批量大小（Batch Size）
    :param n: 每个点云的总点数（Number of points in the point cloud）
    :param m: 采样点的数量（Number of sampled points）
    :param points_tensor: (B, N, 3) 的张量，存储输入点云
    :param temp_tensor: (B, N) 的张量，初始化为 `1e10`，用于存储每个点到最近已选点的距离
    :param idx_tensor: (B, m) 的张量，存储最终采样的点索引
    """
    points = points_tensor.cpu().numpy()  # 转为 NumPy 以便处理
    temp = temp_tensor.cpu().numpy()
    idx = idx_tensor.cpu().numpy()

    B, N, _ = points.shape

    for batch in range(B):
        farthest = 0  # 选择第一个点
        for i in range(m):
            idx[batch, i] = farthest
            centroid = points[batch, farthest, :]
            dist = ((points[batch] - centroid) ** 2).sum(axis=1)  # 计算欧几里得距离的平方
            temp[batch] = np.minimum(temp[batch], dist)  # 更新每个点到已选点集的最短距离
            farthest = temp[batch].argmax()  # 选择最远的点作为下一个采样点

    # 将结果复制回 PyTorch Tensor
    idx_tensor.copy_(torch.from_numpy(idx))


class FurthestPointSampling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        采用最远点采样（Furthest Point Sampling, FPS）来选择 `npoint` 个点，确保采样点集的最小距离最大化。

        :param ctx: PyTorch Autograd 上下文
        :param xyz: (B, N, 3) 的输入张量，表示点云数据，其中 N > npoint
        :param npoint: 需要采样的点数
        :return: (B, npoint) 的张量，存储采样点索引
        """
        assert xyz.is_contiguous(), "xyz 需要是一个连续张量"

        B, N, _ = xyz.shape  # 解析批量大小和点数

        # 初始化索引和距离张量
        output = torch.empty((B, npoint), dtype=torch.int32, device=xyz.device)
        temp = torch.full((B, N), 1e10, dtype=torch.float32, device=xyz.device)

        # 调用 FPS 采样函数
        furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)

        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


def gather_points_kernel_fast(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    使用 Python 实现的 gather_points_kernel_fast
    :param points: (B, C, N) 特征张量
    :param idx: (B, M) 索引张量
    :return: (B, C, M) 采样后的特征
    """
    B, C, N = points.shape
    _, M = idx.shape

    # 创建输出张量
    out = torch.zeros((B, C, M), dtype=torch.float32, device=points.device)

    # 扩展 idx 使其形状适配 features (B, C, M)
    idx_expanded = idx.unsqueeze(1).expand(B, C, M)  # (B, C, M)

    # 使用 gather 从 points 中提取特征，按索引聚合
    out = torch.gather(points, 2, idx_expanded)  # (B, C, M)

    return out, idx_expanded

class GatherOperation(Function):



    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint, device=features.device)

        output = gather_points_kernel_fast(
            B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):    # todo: understand this part. why needs this backward??
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros(
            [B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.gather_points_grad_wrapper(
            B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply
# mark: torch gather is even faster. sampled_xyz = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def fps(data, number):
    '''
        data B N C
        number int
    '''
    fps_idx = furthest_point_sample(data[:, :, :3].contiguous(), number)
    fps_data = torch.gather(
        data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    return fps_data


if __name__ == '__main__':
    import time

    B, C, N = 2, 3, 10000
    K = 16
    device = 'cuda'
    points = torch.randn([B, N, 3], device=device, dtype=torch.float)
    print(points.shape, '\n', points)

    nsample = 4096
    idx = furthest_point_sample(points, nsample)

    st = time.time()
    for _ in range(100):
        query1 = torch.gather(
            points, 1, idx.long().unsqueeze(-1).expand(-1, -1, 3))
    print(time.time() - st)
    print(query1.shape)

    st = time.time()
    for _ in range(100):
        query2 = gather_operation(points.transpose(
            1, 2).contiguous(), idx).transpose(1, 2).contiguous()
    print(time.time() - st)
    print(query2.shape)

    print(torch.allclose(query1, query2))
