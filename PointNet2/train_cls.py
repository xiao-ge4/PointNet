from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse  # 参数存储控制器
import numpy as np  # 数据处理
import os  # 用于获得系统路径、环境
import torch
import datetime
import logging  # 日志打印
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR  # 绝对路径
sys.path.append(os.path.join(ROOT_DIR, 'models'))
"""
需要配置的参数：
--model pointnet2_cls_msg 
--normal 
--log_dir pointnet2_cls_msg
"""

def parse_args():
    ''' PARAMETERS 参数配置 '''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pointnet2_cls_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    # 是否使用法向量信息
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))  # 40*3
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):

    def log_string(str):
        '''
        信息（info）打印
        :param str: 需要打印的内容
        :return: 无返回值
        '''
        logger.info(str)  # 日志端输出
        print(str)  # 程序端输出

    '''HYPER PARAMETER'''
    # 指定所要使用的显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 如果无法使用 cuda ，就改用 cpu ，方法如下
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 其中所有.cuda()的地方都要改为.to(device)

    '''CREATE DIR'''
    # 创建相应文件，存放过程数据
    # 获得时间戳
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    # 先在当前目录（运行目录）下创建/log文件夹
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    # 再在/log文件夹下创建/classification文件夹
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    # 这里默认的设置是None，在命令行可以改变
    if args.log_dir is None:
        # 如果超参数未设置，就在experiment_dir里面加上时间戳，表征本次运行
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        # 如果超参数已设置，就在experiment_dir里面加上自己的超参数标识
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    # 在./log/classification 下增加/${log_dir}/文件
    experiment_dir.mkdir(exist_ok=True)
    # 在./log/classification/${log_dir} 下增加/checkpoints文件
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    # 在./log/classification/${log_dir}下增加/log文件
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    # 参数复制
    args = parse_args()
    # 初始化日志对象
    logger = logging.getLogger("Model")
    # 等级设置（设置日志等级调用比设置等级低的日志记录函数则不会输出）
    logger.setLevel(logging.INFO)
    # 设置输出形式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 设置输出位置
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')  # 输出“PARAMETER ...”
    log_string(args)  # 输出args中存储的参数

    '''DATA LOADING'''
    # 提示信息输出
    log_string('Load dataset ...')
    # 数据路径
    DATA_PATH = 'data/modelnet40_normal_resampled/'

    '''
    ModelNetDataLoader()
    参数
    root 数据存放路径
    npoint 每个 batch 的数据个数 默认：1024
    split 标识是训练还是测试
    normal_channel 是否是 默认：是否使用法向量
    '''
    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',
                                                     normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                    normal_channel=args.normal)
    # torch.utils.data.DataLoader() Pytorch的接口
    # 使用 Pytorch 框架通过这个接口完成输入
    '''
    参数
    原始数据
    batch(8) 多少批数据
    shuffle(Ture) 会在每个epoch重新打乱数据 
    num_workers(4) 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
    '''
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # 模型加载
    # 一共有40个类别
    num_class = 40
    # 将args.model(pointnet2_cls_msg.py) 赋给 MODEL
    MODEL = importlib.import_module(args.model)
    # experiment_dir --> ./log/classification/${log_dir}
    # 这里两句的意思是将./models/下的 pointnet_util.py 复制到 ./log/classification/${log_dir}下
    # 将./models/下的 args.model(pointnet2_cls_msg.py)复制到 ./log/classification/${log_dir}下
    # 开发结束以后直接下载/log文件夹，实验环境主要用于测试
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    # 获得 pointnet2_cls_msg.py 中的两个类
    classifier = MODEL.get_model(num_class,normal_channel=args.normal).to(device)
    criterion = MODEL.get_loss().to(device)

    # 已经有“best_model.pth”
    try:
        # checkpoint <-- ./log/classification/${log_dir}/checkpoints/best_model.pth
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        # 没有“best_model.pth”,从头开始训练数据集
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    # 默认是 Adam
    # 参数优化器
    '''
    参数
    params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
    lr (float, 可选) – 学习率（默认：1e-3）
    betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
    eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
    weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
    '''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        # SGD : 梯度下降法
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0  # 记录总训练次数
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []

    '''TRANING'''
    # 输出开始训练信号
    logger.info('Start training...')
    # start_epoch = 0（上面判断没有初始模型时，设置的）
    # args.epoch = 200 参数加载时设置的
    for epoch in range(start_epoch,args.epoch):
        # 显示进度
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        # 更新优化器学习率
        scheduler.step()
        '''
        enumerate(,)
        目的：将可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        参数：trainDataLoader 可迭代对象 ；0 start，下标开始位置
        迭代的取数据-->ModelNetDataLoader._get_item()
        '''
        '''
        tqdm(,,)
        目的：进度条
        参数：iterable：可迭代对象，total：总的项目数，smoothing：会平均移动因素和预计的时间
        '''
        # TODO  batch_id, data
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # batch_id enumerate(,)返回的下标
            # data 返回的
            points, target = data
            # 数据增强
            # 随机DropOut
            # 缩放
            # 抖动-->加减法
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)  # 维度转换
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()  # 梯度清零

            classifier = classifier.train()
            # 预测
            pred, trans_feat = classifier(points)
            # 误差
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 总训练次数+1
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        # 关于 with 用在：对资源进行访问的场合，过程中是否发生异常都会执行必要的“清理”操作，释放资源
        # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    # 加载参数
    args = parse_args()
    # 进入“main”文件
    main(args)
