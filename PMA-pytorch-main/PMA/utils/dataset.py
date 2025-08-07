"""Data loader
"""
import argparse
import logging
import os
from typing import List
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import torchvision
from sklearn.neighbors import NearestNeighbors

import classification_ModelNet40.utils.transforms as Transforms
# import  src.data_loader.adaptive_voxel_downsampling as  DownSampling
# import src.common.math.se3 as se3
import random
_logger = logging.getLogger()


# def get_train_datasets(args: argparse.Namespace):
#     train_categories, val_categories = None, None
#
#     train_transforms, val_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag,
#                                                       args.num_points, args.partial)
#     # adavoxel_DN = get_AVDN_prossed(args.k, args.base_voxel_size, args.depth_weight, args.curature_weight)
#     _logger.info('Train transforms: {}'.format(', '.join([type(t).__name__ for t in train_transforms])))
#     _logger.info('Val transforms: {}'.format(', '.join([type(t).__name__ for t in val_transforms])))
#     train_transforms = torchvision.transforms.Compose(train_transforms)
#     val_transforms = torchvision.transforms.Compose(val_transforms)
#
#     if args.dataset_type == 'modelnet_hdf':
#         # train_data = XYZDataset(args.dataset_path, subset='train_prossed', transform=train_transforms, pre_pross=adavoxel_DN)
#         train_data = MarginDataset(args.dataset_path, subset='train_prossed', transform=train_transforms)
#
#         val_data = MarginDataset(args.dataset_path, subset='val_prossed', transform=val_transforms)
#     else:
#         raise NotImplementedError
#
#     return train_data, val_data
#
#
# def get_test_datasets(args: argparse.Namespace):
#     test_categories = None
#     if args.test_category_file:
#         test_categories = [line.rstrip('\n') for line in open(args.test_category_file)]
#         test_categories.sort()
#
#     _, test_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag,
#                                         args.num_points, args.partial)
#     # adavoxel_DN = get_AVDN_prossed(args.k, args.base_voxel_size, args.depth_weight, args.curature_weight)
#     _logger.info('Test transforms: {}'.format(', '.join([type(t).__name__ for t in test_transforms])))
#     test_transforms = torchvision.transforms.Compose(test_transforms)
#
#     if args.dataset_type == 'modelnet_hdf':
#         # test_data = XYZDataset(args.dataset_path, subset='val', transform=test_transforms, pre_pross=adavoxel_DN)
#         test_data = MarginDataset(args.dataset_path, subset='test_prossed', transform=test_transforms)
#     else:
#         raise NotImplementedError
#
#     return test_data

def get_transforms(noise_type: str,
                   rot_mag: float = 45.0, trans_mag: float = 0.5,
                   num_points: int = 1024, partial_p_keep: List = None):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        train_transforms = [Transforms.Resampler(num_points),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.FixedResampler(num_points),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomCrop(partial_p_keep),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomCrop(partial_p_keep),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms


class ShellDataset(Dataset):
    def __init__(self, dataset_path: str, subset, npoint=None):
        """Dataset for .xyz point cloud files.

        Args:
            dataset_path (str): Folder containing processed dataset.
            subset (str): Dataset subset, either 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            pre_pross(adaptive_voxel_downsampling):Optional transform to be applied on a sample.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path
        self._train_root = os.path.join(dataset_path, subset)
        self._logger.info('Loading data from {}'.format(dataset_path))

        if not os.path.exists(os.path.join(dataset_path)):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        self.npoint = npoint
        self._file, self._data = self.load_dataset()
        # self._transform = transform
        self._logger.info('Loaded {} {} instances.'.format(len(self._data), subset))

    def process_point_cloud_numpy(self, points):
        # 按 z 轴排序（从小到大）
        sorted_points = points[np.argsort(points[:, 2])]  # 按 z 值升序排列

        # 获取 z 轴的最小值和最大值
        z_min, z_max = sorted_points[:, 2].min(), sorted_points[:, 2].max()

        # 将 z 值分为 6 个等分，得到 7 个分界点
        z_bins = np.linspace(z_min, z_max, 5)  # 7 是因为有 6 份，7 个分界点

        # 去除 z 轴最高的部分，保留 z < z_bins[5] 的点
        mask = (sorted_points[:, 2] < z_bins[3])  # 保留 z 小于 z_bins[5] 的点

        # 保留剩下的 5 部分
        remaining_points = sorted_points[mask]

        return remaining_points

    def remove_far_outliers(self, points, sigma=5):
        """
        使用 3σ 原则剔除远离主区域的点
        :param points: (N, 3) NumPy 数组，点云数据
        :param sigma: 超过 sigma 倍标准差的点将被剔除
        :return: 过滤后的点云
        """
        mean = np.mean(points, axis=0)  # 计算均值
        std_dev = np.std(points, axis=0)  # 计算标准差

        # 过滤掉超出 3 倍标准差范围的点
        mask = np.all(np.abs(points - mean) <= sigma * std_dev, axis=1)
        return points[mask]

    def __getitem__(self, item):
        start_idx = 0
        current_class = None
        for class_label, class_data in self._data.items():
            class_size = len(class_data)
            end_idx = start_idx + class_size
            if start_idx <= item < end_idx:
                current_class = class_label
                idex = item - start_idx
                sample1 = class_data[idex]  # 获取当前类的样本
                class_file = self._file.get(current_class)
                path1 = class_file[idex]
                name1 = os.path.splitext(os.path.basename(path1))[0][:2]
                break
            start_idx = end_idx

        is_same_class = random.choice([0, 1])
        # classes1 = random.choice(list(self._file.keys()))
        if is_same_class:
            # 同类样本对：从当前样本中随机选择一个同类样本（此处没有标签，假设通过索引确定同类）
            # sample1 = self._data.get(current_class)[item]
            idx2 = random.choice([i for i in range(len(self._data.get(current_class))) if i != item])
            path2 = os.path.splitext(os.path.basename(self._file.get(current_class)[idx2]))
            name2 = path2[0][:2]
            sample2 = self._data.get(current_class)[idx2]
        else:
            # sample1 = self._data.get(classes1)[item]
            other_classes = [cls for cls in self._data if cls != current_class]
            classes2 = random.choice(other_classes)
            idx2 = random.choice([i for i in range(len(self._data[classes2]))])
            path2 = os.path.splitext(os.path.basename(self._file.get(classes2)[idx2]))
            name2 = path2[0][:2]
            sample2 = self._data[classes2][idx2]

        # sample1 = self.remove_far_outliers(sample1)
        # sample2 = self.remove_far_outliers(sample2)

        # sample1 = self.process_point_cloud_numpy(sample1)
        # sample2 = self.process_point_cloud_numpy(sample2)

        if self.npoint is not None:
            if sample1.size < self.npoint:
                print(path1)
            if sample2.size < self.npoint:
                print(path2)
            sample1 = self.uniform_sample_point_cloud(sample1, self.npoint)
            sample2 = self.uniform_sample_point_cloud(sample2, self.npoint)

        # sample1 = compute_normals_and_concat(sample1)
        # sample2 = compute_normals_and_concat(sample2)


        # sample = {'sample1': sample1[:, :3], 'sample2': sample2[:, :3], 'name1': name1, 'name2': name2}
        sample = {'sample1': sample1, 'sample2': sample2, 'name1': name1, 'name2': name2}

        # if self._transform:
        #     sample = self._transform(sample)

        return sample, is_same_class

    def uniform_sample_point_cloud(self, point_cloud, m):
        """
        从点云数据中均匀选取指定数量的点。

        :param point_cloud: 输入点云数据，形状为 (n, 3)
        :param m: 要保留的点的数量
        :return: 均匀选取的 m 个点，形状为 (m, 3)
        """
        # 获取点云的总点数
        n = point_cloud.shape[0]

        # 如果 m 大于等于 n，直接返回原始点云
        if m >= n:
            return point_cloud

        # 计算步长，选择均匀间隔的点
        step = n // m
        selected_indices = np.arange(0, n, step)[:m]  # 保证选取 m 个点

        # 从原始点云中选择对应的点
        sampled_points = point_cloud[selected_indices]

        return sampled_points


    # def crop_point(self, point, target_percentage=0.3, target_point_count=None):
    #     """
    #     对点云进行裁剪，保留指定比例的点，并根据目标比例裁剪距离中心的点。
    #
    #     :param point: (n, 6) 形状的点云数据，每个点有 6 个特征（x, y, z, nx, ny, nz）。
    #     :param target_percentage: 目标移除的点的比例，例如 0.3 表示裁剪 30% 的点。
    #     :param target_point_count: 如果指定，强制保留该数量的点。否则使用 target_percentage。
    #     :return: 裁剪后的点云数据。
    #     """
    #     # 1. 从输入的 point (n, 6) 数组中提取坐标数据
    #     point_cloud = point[:, :3]  # 只提取前 3 列作为点云坐标数据
    #     total_points = point_cloud.shape[0]
    #
    #     # 2. 如果提供了目标点数量，使用目标数量进行均匀采样
    #     if target_point_count is not None:
    #         if target_point_count >= total_points:
    #             return point  # 如果目标点数大于或等于总点数，返回原始点云
    #         # 计算均匀采样的步长
    #         step = total_points // target_point_count
    #         selected_indices = np.arange(0, total_points, step)
    #         selected_indices = selected_indices[:target_point_count]  # 确保不会超出
    #         point_cloud = point_cloud[selected_indices]  # 均匀采样后的点云
    #         point = point[selected_indices]  # 保留对应的点云数据和特征
    #
    #     total_points = point_cloud.shape[0]  # 重新计算采样后的点数
    #
    #     # 3. 计算需要移除的点数
    #     target_remove_count = int(total_points * target_percentage)
    #
    #     # 4. 随机选择一个点作为裁剪中心
    #     center_index = np.random.choice(total_points)  # 随机选择一个点的索引
    #     center = point_cloud[center_index]  # 获取该点的坐标
    #
    #     # 5. 计算每个点到中心点的距离
    #     distances = np.linalg.norm(point_cloud - center, axis=1)
    #
    #     # 6. 计算合适的裁剪半径来移除目标数量的点
    #     sorted_distances = np.sort(distances)
    #     radius = sorted_distances[target_remove_count]  # 选择距离第 target_remove_count 个点的距离作为半径
    #
    #     # 7. 筛选出距离大于半径的点（即保留距离大于半径的点，包括法向量）
    #     cropped_point_cloud = point[distances > radius]
    #
    #     return cropped_point_cloud

    def __len__(self):
        # 计算所有文件夹中 .xyz 文件的总数量
        total_files = 0
        for folder_name, paths_list in self._file.items():
            total_files += len(paths_list)  # 累加每个文件夹中 .xyz 文件的数量

        return total_files

    def find_idx_in_data(self, idx, classes):
        # 记录总数，用于累计每个子列表的文件数
        total = 0

        # 遍历 self._data 中的每个子列表
        for i, sublist in enumerate(self._data.values()):
            # 遍历子列表中的每个 ndarray
            length_of_data = len(sublist)  # 当前 ndarray 的数据点数量
            # 判断 idx 是否在当前 data 的范围内
            if total <= idx < total + length_of_data:
                # 计算 idx 在当前 ndarray 中的位置
                position_in_sublist = idx - total - 1
                result_data = sublist[position_in_sublist]
                break
            # 更新 total，继续寻找下一个子列表的数据
            total += length_of_data

        for i, sublist in enumerate(self._file.values()):
            # 遍历子列表中的每个 ndarray
            length_of_data = len(sublist)  # 当前 ndarray 的数据点数量
            # 判断 idx 是否在当前 data 的范围内
            if total <= idx < total + length_of_data:
                # 计算 idx 在当前 ndarray 中的位置
                position_in_sublist = idx - total - 1
                file_path = sublist[position_in_sublist]
                break
            # 更新 total，继续寻找下一个子列表的数据
            total += length_of_data
        # 返回结果数据和文件路径

        return result_data, file_path

    def load_dataset(self):
        # 存储结果的字典，用于动态生成变量
        paths_dict = {}  # 保存文件路径
        data_dict = {}  # 保存文件数据

        # 遍历子文件夹中的所有 .xyz 文件
        for file_name in os.listdir(self._train_root):
            if file_name.endswith('.xyz'):
                file_path = os.path.join(self._train_root, file_name)
                classes = file_name[:2]  # 假设类别信息位于文件名前两位

                # 如果该类别尚未在字典中，初始化列表
                if classes not in paths_dict:
                    paths_dict[classes] = []
                    data_dict[classes] = []

                # 加载 .xyz 文件数据
                data = np.loadtxt(file_path)  # 加载为 NumPy 数组

                # 将文件路径和数据分别添加到字典中对应的类别列表
                paths_dict[classes].append(file_path)
                data_dict[classes].append(data)

        return paths_dict, data_dict


class CustomDataLoader(Dataset):
    def __init__(self, total_folder, index_file, npoint):
        """
        初始化 DataLoader。

        参数：
        - total_folder: 总文件夹路径，包含所有子文件夹。
        - index_file: 索引文件路径，每行是一个组合（如 "folder1_002，folder2_004"）。
        """
        self.npoint = npoint
        self.total_folder = total_folder
        self.index_file = index_file
        self.index_pairs = self._load_index_pairs()

    def _load_index_pairs(self):
        """加载索引文件中的所有组合"""
        with open(self.index_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        return [line.split("，") for line in lines]

    def compute_normals_and_concat(self, xyz, k=30):
        """
        计算点云的法向量，并将其与原始 xyz 数据拼接。

        :param xyz: (N,3) 的 numpy 数组，代表点云的 x, y, z 坐标
        :param k: 计算法向量的 k 近邻数
        :return: (N,6) 的 numpy 数组，包含原始 xyz 和计算出的 nx, ny, nz
        """
        # 转换为 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # 计算法向量
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

        # 获取法向量
        normals = np.asarray(pcd.normals)

        # 拼接 xyz 和法向量
        xyz_with_normals = np.hstack((xyz, normals))

        return xyz_with_normals


    def process_point_cloud_numpy(self, points):
        # 按 z 轴排序（从小到大）
        sorted_points = points[np.argsort(points[:, 2])]  # 按 z 值升序排列

        # 获取 z 轴的最小值和最大值
        z_min, z_max = sorted_points[:, 2].min(), sorted_points[:, 2].max()

        # 将 z 值分为 6 个等分，得到 7 个分界点
        z_bins = np.linspace(z_min, z_max, 5)  # 7 是因为有 6 份，7 个分界点

        # 去除 z 轴最高的部分，保留 z < z_bins[5] 的点
        mask = (sorted_points[:, 2] < z_bins[3])  # 保留 z 小于 z_bins[5] 的点

        # 保留剩下的 5 部分
        remaining_points = sorted_points[mask]

        return remaining_points

    def uniform_sample_point_cloud(self, point_cloud, m):
        """
        从点云数据中均匀选取指定数量的点。

        :param point_cloud: 输入点云数据，形状为 (n, 3)
        :param m: 要保留的点的数量
        :return: 均匀选取的 m 个点，形状为 (m, 3)
        """
        # 获取点云的总点数
        n = point_cloud.shape[0]

        # 如果 m 大于等于 n，直接返回原始点云
        if m >= n:
            return point_cloud

        # 计算步长，选择均匀间隔的点
        step = n // m
        selected_indices = np.arange(0, n, step)[:m]  # 保证选取 m 个点

        # 从原始点云中选择对应的点
        sampled_points = point_cloud[selected_indices]

        return sampled_points

    def _load_data_from_index(self, index):
        """
        根据索引加载数据。

        参数：
        - index: 单个索引（如 "folder1_002"）。

        返回：
        - 数据（如 NumPy 数组）。
        """
        folder_name, suffix = index.split("_")
        folder_path = os.path.join(self.total_folder, folder_name)

        # 遍历子文件夹，找到匹配的文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".xyz"):
                file_suffix = file_name.split("-")[-1].split(".")[0]
                if file_suffix == suffix:
                    file_path = os.path.join(folder_path, file_name)
                    # 假设数据是 NumPy 数组，可以根据实际情况修改
                    data = np.loadtxt(file_path)
                    return data

        raise FileNotFoundError(f"未找到与索引 {index} 匹配的文件")

    def __len__(self):
        """返回数据集的长度"""
        return len(self.index_pairs)

    def apply_transforms(self, point_cloud):
        """对点云数据应用平移、旋转、错切、缩放变换，并加入噪声"""
        # 1. 旋转变换 (绕随机轴)
        axis = np.random.rand(3) - 0.5  # 随机选择旋转轴
        axis /= np.linalg.norm(axis)
        theta = np.random.uniform(0, 2 * np.pi)  # 随机旋转角度
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        ux, uy, uz = axis
        rotation_matrix = np.array([
            [cos_theta + ux ** 2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta,
             ux * uz * (1 - cos_theta) + uy * sin_theta],
            [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy ** 2 * (1 - cos_theta),
             uy * uz * (1 - cos_theta) - ux * sin_theta],
            [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta,
             cos_theta + uz ** 2 * (1 - cos_theta)]
        ])
        point_cloud = np.dot(point_cloud, rotation_matrix.T)

        # 2. 平移变换
        translation = np.random.uniform(-0.2, 0.2, size=(1, 3))
        point_cloud += translation

        # 3. 缩放变换
        scale = np.random.uniform(0.8, 1.2)
        point_cloud *= scale

        # 4. 错切变换 (随机选择两个轴进行错切)
        shear_axes = np.random.choice(3, 2, replace=False)
        shear_matrix = np.eye(3)
        shear_matrix[shear_axes[0], shear_axes[1]] = np.random.uniform(-0.2, 0.2)
        shear_matrix[shear_axes[1], shear_axes[0]] = np.random.uniform(-0.2, 0.2)
        point_cloud = np.dot(point_cloud, shear_matrix.T)

        return point_cloud

    def __getitem__(self, idx):
        """
        根据索引获取一对数据和标签。

        参数：
        - idx: 数据索引。

        返回：
        - data1: 第一个数据。
        - data2: 第二个数据。
        - label: 如果两个数据来自同一个子文件夹返回 1，否则返回 0。
        """
        index1, index2 = self.index_pairs[idx]

        # 提取子文件夹名
        folder1 = index1.split("_")[0]
        folder2 = index2.split("_")[0]

        # 判断是否来自同一个子文件夹
        label = 1 if folder1 == folder2 else 0

        # 加载数据
        data1 = self._load_data_from_index(index1)
        data2 = self._load_data_from_index(index2)
        data1 = self.uniform_sample_point_cloud(data1, self.npoint)
        data2 = self.uniform_sample_point_cloud(data2, self.npoint)

        # data1 = self.compute_normals_and_concat(data1)
        # data2 = self.compute_normals_and_concat(data2)
        sample = {'sample1': data1, 'sample2': data2, 'name1': index1, 'name2': index2}

        return sample, label



class Ransac_Dataset(Dataset):
    def __init__(self, dataset_path: str, subset, npoint=None):
        """Dataset for .xyz point cloud files.

        Args:
            dataset_path (str): Folder containing processed dataset.
            subset (str): Dataset subset, either 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            pre_pross(adaptive_voxel_downsampling):Optional transform to be applied on a sample.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path
        self._train_root = os.path.join(dataset_path, subset)
        self._logger.info('Loading data from {}'.format(dataset_path))

        if not os.path.exists(os.path.join(dataset_path)):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        self.npoint = npoint
        self._file, self._data = self.load_dataset()
        # self._transform = transform
        self._logger.info('Loaded {} {} instances.'.format(len(self._data), subset))

    def __getitem__(self, item):
        start_idx = 0
        current_class = None
        for class_label, class_data in self._data.items():
            class_size = len(class_data)
            end_idx = start_idx + class_size
            if start_idx <= item < end_idx:
                current_class = class_label
                idex = item - start_idx
                sample1 = class_data[idex]  # 获取当前类的样本
                class_file = self._file.get(current_class)
                path1 = class_file[idex]
                name1 = os.path.splitext(os.path.basename(path1))[0][:2]
                break
            start_idx = end_idx

        is_same_class = random.choice([0, 1])
        # classes1 = random.choice(list(self._file.keys()))
        if is_same_class:
            # 同类样本对：从当前样本中随机选择一个同类样本（此处没有标签，假设通过索引确定同类）
            # sample1 = self._data.get(current_class)[item]
            idx2 = random.choice([i for i in range(len(self._data.get(current_class))) if i != item])
            name2 = os.path.splitext(os.path.basename(self._file.get(current_class)[idx2]))[0][:2]
            sample2 = self._data.get(current_class)[idx2]
        else:
            # sample1 = self._data.get(classes1)[item]
            other_classes = [cls for cls in self._data if cls != current_class]
            classes2 = random.choice(other_classes)
            idx2 = random.choice([i for i in range(len(self._data[classes2]))])
            name2 = os.path.splitext(os.path.basename(self._file.get(classes2)[idx2]))[0][:2]
            sample2 = self._data[classes2][idx2]

        if self.npoint is not None:
            sample1 = self.uniform_sample_point_cloud(sample1, self.npoint)
            sample2 = self.uniform_sample_point_cloud(sample2, self.npoint)

        sample1 = compute_normals_and_concat(sample1)
        sample2 = compute_normals_and_concat(sample2)

        # sample = {'sample1': sample1[:, :3], 'sample2': sample2[:, :3], 'name1': name1, 'name2': name2}
        sample = {'sample1': sample1, 'sample2': sample2, 'name1': name1, 'name2': name2}

        # if self._transform:
        #     sample = self._transform(sample)

        return sample, is_same_class

    def uniform_sample_point_cloud(self, point_cloud, m):
        """
        从点云数据中均匀选取指定数量的点。

        :param point_cloud: 输入点云数据，形状为 (n, 3)
        :param m: 要保留的点的数量
        :return: 均匀选取的 m 个点，形状为 (m, 3)
        """
        # 获取点云的总点数
        n = point_cloud.shape[0]

        # 如果 m 大于等于 n，直接返回原始点云
        if m >= n:
            return point_cloud

        # 计算步长，选择均匀间隔的点
        step = n // m
        selected_indices = np.arange(0, n, step)[:m]  # 保证选取 m 个点

        # 从原始点云中选择对应的点
        sampled_points = point_cloud[selected_indices]

        return sampled_points


    # def crop_point(self, point, target_percentage=0.3, target_point_count=None):
    #     """
    #     对点云进行裁剪，保留指定比例的点，并根据目标比例裁剪距离中心的点。
    #
    #     :param point: (n, 6) 形状的点云数据，每个点有 6 个特征（x, y, z, nx, ny, nz）。
    #     :param target_percentage: 目标移除的点的比例，例如 0.3 表示裁剪 30% 的点。
    #     :param target_point_count: 如果指定，强制保留该数量的点。否则使用 target_percentage。
    #     :return: 裁剪后的点云数据。
    #     """
    #     # 1. 从输入的 point (n, 6) 数组中提取坐标数据
    #     point_cloud = point[:, :3]  # 只提取前 3 列作为点云坐标数据
    #     total_points = point_cloud.shape[0]
    #
    #     # 2. 如果提供了目标点数量，使用目标数量进行均匀采样
    #     if target_point_count is not None:
    #         if target_point_count >= total_points:
    #             return point  # 如果目标点数大于或等于总点数，返回原始点云
    #         # 计算均匀采样的步长
    #         step = total_points // target_point_count
    #         selected_indices = np.arange(0, total_points, step)
    #         selected_indices = selected_indices[:target_point_count]  # 确保不会超出
    #         point_cloud = point_cloud[selected_indices]  # 均匀采样后的点云
    #         point = point[selected_indices]  # 保留对应的点云数据和特征
    #
    #     total_points = point_cloud.shape[0]  # 重新计算采样后的点数
    #
    #     # 3. 计算需要移除的点数
    #     target_remove_count = int(total_points * target_percentage)
    #
    #     # 4. 随机选择一个点作为裁剪中心
    #     center_index = np.random.choice(total_points)  # 随机选择一个点的索引
    #     center = point_cloud[center_index]  # 获取该点的坐标
    #
    #     # 5. 计算每个点到中心点的距离
    #     distances = np.linalg.norm(point_cloud - center, axis=1)
    #
    #     # 6. 计算合适的裁剪半径来移除目标数量的点
    #     sorted_distances = np.sort(distances)
    #     radius = sorted_distances[target_remove_count]  # 选择距离第 target_remove_count 个点的距离作为半径
    #
    #     # 7. 筛选出距离大于半径的点（即保留距离大于半径的点，包括法向量）
    #     cropped_point_cloud = point[distances > radius]
    #
    #     return cropped_point_cloud

    def __len__(self):
        # 计算所有文件夹中 .xyz 文件的总数量
        total_files = 0
        for folder_name, paths_list in self._file.items():
            total_files += len(paths_list)  # 累加每个文件夹中 .xyz 文件的数量

        return total_files

    def find_idx_in_data(self, idx, classes):
        # 记录总数，用于累计每个子列表的文件数
        total = 0

        # 遍历 self._data 中的每个子列表
        for i, sublist in enumerate(self._data.values()):
            # 遍历子列表中的每个 ndarray
            length_of_data = len(sublist)  # 当前 ndarray 的数据点数量
            # 判断 idx 是否在当前 data 的范围内
            if total <= idx < total + length_of_data:
                # 计算 idx 在当前 ndarray 中的位置
                position_in_sublist = idx - total - 1
                result_data = sublist[position_in_sublist]
                break
            # 更新 total，继续寻找下一个子列表的数据
            total += length_of_data

        for i, sublist in enumerate(self._file.values()):
            # 遍历子列表中的每个 ndarray
            length_of_data = len(sublist)  # 当前 ndarray 的数据点数量
            # 判断 idx 是否在当前 data 的范围内
            if total <= idx < total + length_of_data:
                # 计算 idx 在当前 ndarray 中的位置
                position_in_sublist = idx - total - 1
                file_path = sublist[position_in_sublist]
                break
            # 更新 total，继续寻找下一个子列表的数据
            total += length_of_data
        # 返回结果数据和文件路径

        return result_data, file_path

    def load_dataset(self):
        # 存储结果的字典，用于动态生成变量
        paths_dict = {}  # 保存文件路径
        data_dict = {}  # 保存文件数据

        # 遍历子文件夹中的所有 .xyz 文件
        for file_name in os.listdir(self._train_root):
            if file_name.endswith('.xyz'):
                file_path = os.path.join(self._train_root, file_name)
                classes = file_name[:2]  # 假设类别信息位于文件名前两位

                # 如果该类别尚未在字典中，初始化列表
                if classes not in paths_dict:
                    paths_dict[classes] = []
                    data_dict[classes] = []

                # 加载 .xyz 文件数据
                data = o3d.io.read_point_cloud(file_path)  # 加载为 NumPy 数组

                # 将文件路径和数据分别添加到字典中对应的类别列表
                paths_dict[classes].append(file_path)
                data_dict[classes].append(data)

        return paths_dict, data_dict


class TripleDataset(Dataset):
    def __init__(self, dataset_path: str, subset, npoint=None):
        """Dataset for .xyz point cloud files.

        Args:
            dataset_path (str): Folder containing processed dataset.
            subset (str): Dataset subset, either 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            pre_pross(adaptive_voxel_downsampling):Optional transform to be applied on a sample.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path
        self._train_root = os.path.join(dataset_path, subset)
        self._logger.info('Loading data from {}'.format(dataset_path))

        if not os.path.exists(os.path.join(dataset_path)):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        self.npoint = npoint
        self._file, self._data = self.load_dataset()
        # self._transform = transform
        self._logger.info('Loaded {} {} instances.'.format(len(self._data), subset))

    def __getitem__(self, item):
        start_idx = 0
        current_class = None
        for class_label, class_data in self._data.items():
            class_size = len(class_data)
            end_idx = start_idx + class_size
            if start_idx <= item < end_idx:
                current_class = class_label
                idex = item - start_idx
                anchor = class_data[idex]  # 获取当前类的样本
                class_file = self._file.get(current_class)
                path1 = class_file[idex]
                anchor_name = os.path.splitext(os.path.basename(path1))[0][:2]
                break
            start_idx = end_idx

        is_same_class = random.choice([0, 1])
        # classes1 = random.choice(list(self._file.keys()))
        # if is_same_class:
            # 同类样本对：从当前样本中随机选择一个同类样本（此处没有标签，假设通过索引确定同类）
            # sample1 = self._data.get(current_class)[item]
        idx2 = random.choice([i for i in range(len(self._data.get(current_class))) if i != item])
        positive_name = os.path.splitext(os.path.basename(self._file.get(current_class)[idx2]))[0][:2]
        positive = self._data.get(current_class)[idx2]
        # else:
            # sample1 = self._data.get(classes1)[item]
        other_classes = [cls for cls in self._data if cls != current_class]
        classes2 = random.choice(other_classes)
        idx2 = random.choice([i for i in range(len(self._data[classes2]))])
        negative_name = os.path.splitext(os.path.basename(self._file.get(classes2)[idx2]))[0][:2]
        negative = self._data[classes2][idx2]

        if self.npoint is not None:
            anchor = self.uniform_sample_point_cloud(anchor, self.npoint)
            positive = self.uniform_sample_point_cloud(positive, self.npoint)
            negative = self.uniform_sample_point_cloud(negative, self.npoint)

        # anchor = compute_normals_and_concat(anchor)
        # positive = compute_normals_and_concat(positive)
        # negative = compute_normals_and_concat(negative)

        # sample = {'anchor': anchor[:, :3], 'positive': positive[:, :3], 'negative': negative[:, :3],
        #           'anchor_name': anchor_name, 'positive_name': positive_name, 'negative_name':negative_name}

        sample = {'anchor': anchor, 'positive': positive, 'negative': negative,
                  'anchor_name': anchor_name, 'positive_name': positive_name, 'negative_name': negative_name}

        # if self._transform:
        #     sample = self._transform(sample)

        return sample

    def uniform_sample_point_cloud(self, point_cloud, m):
        """
        从点云数据中均匀选取指定数量的点。

        :param point_cloud: 输入点云数据，形状为 (n, 3)
        :param m: 要保留的点的数量
        :return: 均匀选取的 m 个点，形状为 (m, 3)
        """
        # 获取点云的总点数
        n = point_cloud.shape[0]

        # 如果 m 大于等于 n，直接返回原始点云
        if m >= n:
            return point_cloud

        # 计算步长，选择均匀间隔的点
        step = n // m
        selected_indices = np.arange(0, n, step)[:m]  # 保证选取 m 个点

        # 从原始点云中选择对应的点
        sampled_points = point_cloud[selected_indices]

        return sampled_points


    # def crop_point(self, point, target_percentage=0.3, target_point_count=None):
    #     """
    #     对点云进行裁剪，保留指定比例的点，并根据目标比例裁剪距离中心的点。
    #
    #     :param point: (n, 6) 形状的点云数据，每个点有 6 个特征（x, y, z, nx, ny, nz）。
    #     :param target_percentage: 目标移除的点的比例，例如 0.3 表示裁剪 30% 的点。
    #     :param target_point_count: 如果指定，强制保留该数量的点。否则使用 target_percentage。
    #     :return: 裁剪后的点云数据。
    #     """
    #     # 1. 从输入的 point (n, 6) 数组中提取坐标数据
    #     point_cloud = point[:, :3]  # 只提取前 3 列作为点云坐标数据
    #     total_points = point_cloud.shape[0]
    #
    #     # 2. 如果提供了目标点数量，使用目标数量进行均匀采样
    #     if target_point_count is not None:
    #         if target_point_count >= total_points:
    #             return point  # 如果目标点数大于或等于总点数，返回原始点云
    #         # 计算均匀采样的步长
    #         step = total_points // target_point_count
    #         selected_indices = np.arange(0, total_points, step)
    #         selected_indices = selected_indices[:target_point_count]  # 确保不会超出
    #         point_cloud = point_cloud[selected_indices]  # 均匀采样后的点云
    #         point = point[selected_indices]  # 保留对应的点云数据和特征
    #
    #     total_points = point_cloud.shape[0]  # 重新计算采样后的点数
    #
    #     # 3. 计算需要移除的点数
    #     target_remove_count = int(total_points * target_percentage)
    #
    #     # 4. 随机选择一个点作为裁剪中心
    #     center_index = np.random.choice(total_points)  # 随机选择一个点的索引
    #     center = point_cloud[center_index]  # 获取该点的坐标
    #
    #     # 5. 计算每个点到中心点的距离
    #     distances = np.linalg.norm(point_cloud - center, axis=1)
    #
    #     # 6. 计算合适的裁剪半径来移除目标数量的点
    #     sorted_distances = np.sort(distances)
    #     radius = sorted_distances[target_remove_count]  # 选择距离第 target_remove_count 个点的距离作为半径
    #
    #     # 7. 筛选出距离大于半径的点（即保留距离大于半径的点，包括法向量）
    #     cropped_point_cloud = point[distances > radius]
    #
    #     return cropped_point_cloud

    def __len__(self):
        # 计算所有文件夹中 .xyz 文件的总数量
        total_files = 0
        for folder_name, paths_list in self._file.items():
            total_files += len(paths_list)  # 累加每个文件夹中 .xyz 文件的数量

        return total_files

    def find_idx_in_data(self, idx, classes):
        # 记录总数，用于累计每个子列表的文件数
        total = 0

        # 遍历 self._data 中的每个子列表
        for i, sublist in enumerate(self._data.values()):
            # 遍历子列表中的每个 ndarray
            length_of_data = len(sublist)  # 当前 ndarray 的数据点数量
            # 判断 idx 是否在当前 data 的范围内
            if total <= idx < total + length_of_data:
                # 计算 idx 在当前 ndarray 中的位置
                position_in_sublist = idx - total - 1
                result_data = sublist[position_in_sublist]
                break
            # 更新 total，继续寻找下一个子列表的数据
            total += length_of_data

        for i, sublist in enumerate(self._file.values()):
            # 遍历子列表中的每个 ndarray
            length_of_data = len(sublist)  # 当前 ndarray 的数据点数量
            # 判断 idx 是否在当前 data 的范围内
            if total <= idx < total + length_of_data:
                # 计算 idx 在当前 ndarray 中的位置
                position_in_sublist = idx - total - 1
                file_path = sublist[position_in_sublist]
                break
            # 更新 total，继续寻找下一个子列表的数据
            total += length_of_data
        # 返回结果数据和文件路径

        return result_data, file_path

    def load_dataset(self):
        # 存储结果的字典，用于动态生成变量
        paths_dict = {}  # 保存文件路径
        data_dict = {}  # 保存文件数据

        # 遍历子文件夹中的所有 .xyz 文件
        for file_name in os.listdir(self._train_root):
            if file_name.endswith('.xyz'):
                file_path = os.path.join(self._train_root, file_name)
                classes = file_name[:2]  # 假设类别信息位于文件名前两位

                # 如果该类别尚未在字典中，初始化列表
                if classes not in paths_dict:
                    paths_dict[classes] = []
                    data_dict[classes] = []

                # 加载 .xyz 文件数据
                data = np.loadtxt(file_path)  # 加载为 NumPy 数组

                # 将文件路径和数据分别添加到字典中对应的类别列表
                paths_dict[classes].append(file_path)
                data_dict[classes].append(data)

        return paths_dict, data_dict

def compute_normals_and_concat(xyz, k=30):
    """
    计算点云的法向量，并将其与原始 xyz 数据拼接。

    :param xyz: (N,3) 的 numpy 数组，代表点云的 x, y, z 坐标
    :param k: 计算法向量的 k 近邻数
    :return: (N,6) 的 numpy 数组，包含原始 xyz 和计算出的 nx, ny, nz
    """
    # 转换为 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # 计算法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

    # 获取法向量
    normals = np.asarray(pcd.normals)

    # 拼接 xyz 和法向量
    xyz_with_normals = np.hstack((xyz, normals))

    return xyz_with_normals