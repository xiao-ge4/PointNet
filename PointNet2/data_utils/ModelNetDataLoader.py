import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')



def pc_normalize(pc):
    # 点集标准化
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root  # 'data/modelnet40_normal_resampled/'
        self.npoints = npoint  # 1024
        # 标识是否最远点采样
        self.uniform = uniform  # 默认False
        # 在  self.root = 'data/modelnet40_normal_resampled/'的基础上添加路径 'modelnet40_shape_names.txt'
        # 指向形状存放的txt文件
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        # 将类别划分开，此时 self.cat 获得的是一个数组，共40个元素，每个元素是一个类别的名称
        self.cat = [line.rstrip() for line in open(self.catfile)]
        '''
        函数说明：这里使用了四个函数
        len() 列表长度 40
        range() 循环40次
        zip() 将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
              在这里将 self.cat 的每个元素与 range() 的每个元素打包成元组，相当于让每个类别有对应的序号
        dict() 将 zip() 后的对象变成键值对的形式如：{'bathtub': 0 ',bed': 1}
        
        最终 self.classes 中存储的就是{'bathtub': 0 ',bed': 1,......}
        '''
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel  # 默认True，但是他传入的函数应该是 False 即不使用法向量信息

        shape_ids = {}
        # 将所有数据对应的标签取进来(train & test都取)
        # 注意这里就不是一个标签了，比如飞机这个物体，就有很多模型，对应很多标签（飞机一号、飞机二号......）
        # 结果是 shape_ids['train'] 是9843*1的列表;shape_ids['test'] 是2468*1的列表
        # 而 shape_ids 是一个2*1的列表
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        # 断言：判断是否满足条件（在‘train’中或者’test‘中），如果两者都不是，直接结束
        assert (split == 'train' or split == 'test')
        # 在 train 或者 test 中（取决于“split”参数），将现在所有的标签去掉"_"符号，以及后缀
        # 即“飞机一号”变成“飞机”，例如：bathtub_0110-->bathtub
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # 制作关于元组（标签，图片地址）的列表
        # 结果是(标签，“root，标签，shape_ids[split][i]”)
        # 例如：("airplane",'data/modelnet40_normal_resampled/airplane/airplane_0001')
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        # 输出 self.datapath 的数据大小 9843 * 2
        print('The size of %s data is %d'%(split,len(self.datapath)))

        # how many data points to cache in memory
        # 默认15000
        # 缓存中记忆多少数据
        self.cache_size = cache_size
        # from index to (point_set, cls) tuple
        # 已经存储的记忆
        self.cache = {}

    def __len__(self):
        # ModelNetDataLoader 的数据长度是"self.datapath"参数的长度
        return len(self.datapath)

    def _get_item(self, index):

        '''
        根据索引查找数据具体信息
        :param index:
        :return: point_set:坐标点(xyz)  cls:具体类别
        '''

        # 现在缓存里面查找
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            # 文件路径txt
            fn = self.datapath[index]
            # 加载类别数字
            cls = self.classes[self.datapath[index][0]]
            # 加载类别
            cls = np.array([cls]).astype(np.int32)
            # 坐标点（xyz + 额外特征）
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            # 采样
            if self.uniform:
                # 最远点
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                # 前 npoints 个点
                point_set = point_set[0:self.npoints,:]

            # 现在的点-->1024*6

            # 坐标前三个值 xyz 标准化
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])# 相当于3列值，做标准化

            # 如果不需要法向量
            # 则只返回数据前三列（xyz）
            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            # 取到的点放入缓存中
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)




if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)