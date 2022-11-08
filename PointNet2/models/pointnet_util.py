import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    输入点的索引，返回点的真实坐标
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
        返回中心点
    """
    # 获得设备
    device = xyz.device
    B, N, C = xyz.shape
    # 初始化 8*512
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # 距离初始化 8*1024
    distance = torch.ones(B, N).to(device) * 1e10
    # 最远点初始化 batch里每个样本随机初始化一个最远点的索引
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # 以 batch 为长度的list
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        # 循环 中心点 次（每次找一个点）
        # 8个batch 同时计算
        centroids[:, i] = farthest   # 第一个采样点选随机初始化的索引
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # 得到当前采样点的坐标 B*3
        dist = torch.sum((xyz - centroid) ** 2, -1)  # 计算当前采样点与其他点的距离
        mask = dist < distance  # 选择距离最近的来更新距离（更新维护这个表）
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]  # 重新计算得到最远点索引（在更新的表中选择距离最大的那个点）
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # .view() 类似于resize
    # repeat 重复次数，实际上就是增加维度
    # 初始化结果 group_idx-->[B,S,N]
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # （1024 与每个 512中的点的距离）* 8
    sqrdists = square_distance(new_xyz, xyz)
    # 找到距离大于给定半径的设置成一个N值（1024）索引
    group_idx[sqrdists > radius ** 2] = N
    # 做升序排序，后面的都是大的值（1024）
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # 如果半径内的点没那么多，就直接用第一个点来代替了
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # mask 用来标识不在半径范围内的点
    mask = group_idx == N
    # 把最近点来填充
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        print(xyz.shape)
        if points is not None:
            points = points.permute(0, 2, 1) # 维度转换
        print(points.shape)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        print(new_points.shape) #[8*1*128*643]
        # 维度转换
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        print(new_points.shape)
        # 卷积 3次mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        print(new_points.shape)
        new_points = torch.max(new_points, 2)[0]
        print(new_points.shape)
        new_xyz = new_xyz.permute(0, 2, 1)
        print(new_xyz.shape)
        # 输出坐标 + points 标识
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N] 位置信息
            points: input points data, [B, D, N] 特征信息（init:3*法向量）
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S] #concat理解：特征拼接
        """
        xyz = xyz.permute(0, 2, 1) # 交换点的维度-->矩阵变换（维度1，2进行变换）
        print(xyz.shape)  # [8*3*1024]-->[8*1024*3]
        if points is not None:
            points = points.permute(0, 2, 1)  # 就是额外提取的特征的维度变换，第一次的时候就是那个法向量特征
        print(points.shape)
        B, N, C = xyz.shape  # B:batch N:点的数量 C:坐标维度
        S = self.npoint  # 中心点数量
        # 最远点采样 farthest_point_sample(xyz, S) 坐标xyz+采样数量S
        # index_points()
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))#采样后的点
        # [8*512*3]
        print(new_xyz.shape)
        # group
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            # 遍历半径
            # K为固定半径对应的数据点数量
            K = self.nsample_list[i]
            '''
            参数
            radius 半径
            K group里点的数量
            xyz 1024 所有点
            new_xyz 512 中心点
            返回
            索引
            '''
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            # 得到各个组中实际点
            grouped_xyz = index_points(xyz, group_idx)
            # 去mean new_xyz相当于簇的中心点
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                # 拼接法向量
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
                print(grouped_points.shape)
            else:
                grouped_points = grouped_xyz

            # channel First 维度转换
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            print(grouped_points.shape)
            # 卷积
            for j in range(len(self.conv_blocks[i])):
                # 第一次卷积
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            print(grouped_points.shape)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S] 就是pointnet里的maxpool操作
            print(new_points.shape)
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        print(new_points_concat.shape)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        print(xyz1.shape)
        print(xyz2.shape)

        points2 = points2.permute(0, 2, 1)
        print(points2.shape)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
            print(interpolated_points.shape)
        else:
            dists = square_distance(xyz1, xyz2)
            print(dists.shape)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            print(weight.shape)
            print(index_points(points2, idx).shape)
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            print(interpolated_points.shape)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        print(new_points.shape)
        new_points = new_points.permute(0, 2, 1)
        print(new_points.shape)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        print(new_points.shape)
        return new_points

