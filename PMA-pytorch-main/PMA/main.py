"""
Usage:
python main.py --model PointMLP --msg demo
"""

import ocnn
import dwconv
import argparse
import os
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, random_split
import models as models
from classification_ModelNet40.utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
# from data import ModelNet40
from utils.dataset import ShellDataset, compute_normals_and_concat, CustomDataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--data_root', default='../dataset/shell/prime/normal/', help='path to dataset')
    parser.add_argument('--total_folder', default='../dataset/shell/prime/normal/', help='path to dataset')
    parser.add_argument('--index_file_train_same', default='../dataset/shell/prime/normal/train_set_same.txt', help='path to dataset')
    parser.add_argument('--index_file_train_diff', default='../dataset/shell/prime/normal/train_set_diff.txt', help='path to dataset')
    parser.add_argument('--index_file_test_same', default='../dataset/shell/prime/normal/test_set_same.txt',
                        help='path to dataset')
    parser.add_argument('--index_file_test_diff', default='../dataset/shell/prime/normal/test_set_diff.txt',
                        help='path to dataset')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--model', default='PMA', help='model name [default: pointMLP, pointMLPElite]')
    parser.add_argument('--epoch', default=402, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--val_freq', default=1, type=int, help= 'val_frequence')
    parser.add_argument('--min_lr', default=0.001, type=float, help='min lr')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='workers')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # os.environ['PYTHONUNBUFFERED'] = '1'

    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
    device = 'cuda'
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(10)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg
    args.checkpoint = 'checkpoints/' + args.model + message + '-' + str(args.seed)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    # Model
    printf(f"args: {args}")
    printf('==> Building model..')
    net = models.__dict__[args.model]()
    criterion = models.ContrastiveLoss(margin=10.0)
    net = net.to(device)
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    best_test_acc = 0.  # best test accuracy
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    args.checkpoint = '../checkpoints/DGCNN/'

    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
        train_names = ["Epoch-Num", 'Learning-Rate', 'Train-Loss', 'Train-acc-B', 'Train-acc']
        valid_names = ["Epoch-Num", 'Learning-Rate', 'Valid-Loss', 'Valid-acc-B', 'Valid-acc']

    else:
        printf(f"Resuming last checkpoint from {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model, resume=True)
        optimizer_dict = checkpoint['optimizer']
        train_names = ["Epoch-Num", 'Learning-Rate', 'Train-Loss', 'Train-acc-B', 'Train-acc']
        valid_names = ["Epoch-Num", 'Learning-Rate', 'Valid-Loss', 'Valid-acc-B', 'Valid-acc']


    printf('==> Preparing data..')
    # train_dataset = ShellDataset(args.data_root, subset='train', npoint=args.num_points)
    # test_dataset = ShellDataset(args.data_root, subset='test', npoint=args.num_points)
    # train_dataset_diff = CustomDataLoader(args.total_folder, args.index_file_train_diff, args.num_points)
    # train_dataset_same = CustomDataLoader(args.total_folder, args.index_file_train_same, args.num_points)
    # test_dataset_diff = CustomDataLoader(args.total_folder, args.index_file_test_diff, args.num_points)
    # test_dataset_same = CustomDataLoader(args.total_folder, args.index_file_test_same, args.num_points)
    # train_dataset = torch.utils.data.ConcatDataset([train_dataset_diff, train_dataset_same])
    # test_dataset = torch.utils.data.ConcatDataset([test_dataset_diff, test_dataset_same])
    train_dataset_diff = CustomDataLoader(args.total_folder, args.index_file_train_diff, args.num_points)
    train_dataset_same = CustomDataLoader(args.total_folder, args.index_file_train_same, args.num_points)
    test_dataset_diff = CustomDataLoader(args.total_folder, args.index_file_test_diff, args.num_points)
    test_dataset_same = CustomDataLoader(args.total_folder, args.index_file_test_same, args.num_points)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_diff, train_dataset_same])
    test_dataset = torch.utils.data.ConcatDataset([test_dataset_diff, test_dataset_same])
    # # 获取数据集的总长度
    # full_dataset_len = len(full_dataset)
    #
    # # 按照 3:7 的比例划分数据集
    # train_size = int(0.7 * full_dataset_len)  # 30% 用于训练
    # test_size = full_dataset_len - train_size  # 剩余 70% 用于测试
    #
    # # 随机打乱并进行划分
    # train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr, last_epoch=start_epoch - 1)

    writer = SummaryWriter(log_dir=args.checkpoint)

    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(epoch, net, train_loader, optimizer, criterion, device, logger, writer)  # {"loss", "acc", "acc_avg", "time"}

        scheduler.step()

        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss
        logger.set_names(train_names)
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc_avg"], train_out["acc"]])

        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")

        if epoch % args.val_freq == 0:
            test_out = validate(epoch, net, test_loader, criterion, device, logger, writer)

            if test_out["acc"] > best_test_acc:
                best_test_acc = test_out["acc"]
                is_best = True
            else:
                is_best = False

            best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
            best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
            best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss

            save_model(
                net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
                best_test_acc=best_test_acc,  # best test accuracy
                best_train_acc=best_train_acc,
                best_test_acc_avg=best_test_acc_avg,
                best_train_acc_avg=best_train_acc_avg,
                best_test_loss=best_test_loss,
                best_train_loss=best_train_loss,
                optimizer=optimizer.state_dict()
            )
            logger.set_names(valid_names)
            logger.append([epoch, optimizer.param_groups[0]['lr'],
                           test_out["loss"], test_out["acc_avg"], test_out["acc"]])
            printf(
                f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
                f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")


    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    printf(f"++++++++" * 5)


import numpy as np
import torch


def find_best_threshold(distances, labels, threshold_range=None, step=0.1):

    distances = distances.cpu().numpy() if isinstance(distances, torch.Tensor) else distances
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    if threshold_range is None:
        min_thresh, max_thresh = distances.min(), distances.max()
    else:
        min_thresh, max_thresh = threshold_range

    best_threshold = min_thresh
    best_accuracy = 0

    for threshold in np.arange(min_thresh, max_thresh, step):
        preds = (distances < threshold).astype(int)
        accuracy = np.mean(preds == labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold


def train(epoch, net, trainloader, optimizer, criterion, device, logger, writer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    corrects = 0
    equal_counts = 0
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in enumerate(trainloader):
        data['sample1'], data['sample2'], label = data['sample1'].to(device), data['sample2'].to(device), label.to(
            device).squeeze()
        data['sample1'] = data['sample1'].permute(0, 2, 1).float()
        data['sample2'] = data['sample2'].permute(0, 2, 1).float()  # so, the input data shape is [batch, 3, 1024]
        # data['sample1'] = data['sample1'].float()
        # data['sample2'] = data['sample2'].float()  # so, the input data shape is [batch, 3, 1024]
        logits1 = net(data['sample1'])
        logits2 = net(data['sample2'])
        euclidean_distance = F.pairwise_distance(logits1, logits2)
        # correct = 1 / (1 + euclidean_distance)
        pred = torch.where(euclidean_distance < 20, 1, 0)
        loss = criterion(euclidean_distance, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        train_loss += loss.item()

        # writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(trainloader) + batch_idx)
        # writer.add_scalar('label/train_batch', label.item(), epoch * len(trainloader) + batch_idx)
        # writer.add_scalar('euclidean_distance/train_batch', euclidean_distance, epoch * len(trainloader) + batch_idx)

        train_true.append(label.cpu().numpy())

        train_pred.append(pred.detach().cpu().numpy())

        total += label.size(0)
        equal_count = sum(1 if a == b else 0 for a, b in zip(label, pred))
        equal_counts += equal_count

        for val1, val2, p, q, dist in zip(data['name1'], data['name2'], label.float(), pred.float(), euclidean_distance.float()):
            logger.info("%s and %s label %s pred %s dist %s", val1, val2, p.float(), q.float(), dist.float())
            print(f"{val1} and {val2} label {p.float()} pred {q.float()} dist {dist.float()}")

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * equal_counts / total, equal_counts, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(epoch, net, testloader, criterion, device, logger, writer):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    equal_counts = 0
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data['sample1'], data['sample2'], label = data['sample1'].to(device), data['sample2'].to(device), label.to(
                device).squeeze()
            data['sample1'] = data['sample1'].permute(0, 2, 1).float()
            data['sample2'] = data['sample2'].permute(0, 2, 1).float()  # so, the input data shape is [batch, 3, 1024]
            logits1, logits2 = net(data['sample1'], data['sample2'])
            euclidean_distance = F.pairwise_distance(logits1, logits2)
            # correct = 1 / (1 + euclidean_distance)
            pred = torch.where(euclidean_distance < 20, 1, 0)
            loss = criterion(euclidean_distance, label, pred)

            # writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(testloader) + batch_idx)
            # writer.add_scalar('label/train_batch', label.item(), epoch * len(testloader) + batch_idx)
            # writer.add_scalar('euclidean_distance/train_batch', euclidean_distance,
            #                   epoch * len(testloader) + batch_idx)

            test_loss += loss.item()
            # preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())

            test_pred.append(pred.detach().cpu().numpy())
            total += label.size(0)
            equal_count = sum(1 if a == b else 0 for a, b in zip(label, pred))
            equal_counts += equal_count
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * equal_counts / total, equal_counts, total))

            for val1, val2, p, q, dist in zip(data['name1'], data['name2'], label.float(), pred.float(),
                                              euclidean_distance.float()):
                logger.info("%s and %s label %s pred %s dist %s", val1, val2, p.float(), q.float(), dist.float())
                print(f"{val1} and {val2} label {p.float()} pred {q.float()} dist {dist.float()}")

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


if __name__ == '__main__':
    main()
