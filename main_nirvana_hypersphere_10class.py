#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import datetime
import os
import time
import json
import numpy as np
import torch
import modules.utils_torchvision as utils
from modules.NirvanaLoss import (
    center_loss_nirvana,
    accuracy_l2_nosubcenter,
    get_l2_pred_nosubcenter,
    cosine_similarity,
    euc_cos,
    nirvana_mics_loss,
    nirvana_hypersphere,
)
from torch.utils.tensorboard import SummaryWriter
from modules.lenet import LeNet
from torchvision import transforms
from matplotlib import pyplot as plt
from modules.utils_mine import plot_features
import warnings
from datasets.osr_dataloader import CIFAR10_OSR

# torch.autograd.set_detect_anomaly(True)
# import modules.resnet2 as resnet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = str(4)
warnings.filterwarnings("ignore")


def train_one_epoch(
    model,
    centerloss,
    optimizer,
    optimizer_center,
    data_loader,
    device,
    epoch,
    args,
    debug_param,
):
    model.train(), centerloss.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value:.1f}"))
    header = "Epoch: [{}]".format(epoch)
    all_features, all_labels = [], []
    epoch_loss = 0.0
    for image, target in metric_logger.log_every(data_loader, args.print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        if args.distributed:
            centers = centerloss.module.centers
        else:
            centers = centerloss.centers

        feats, _ = model(image)
        intraclass_loss, triplet_loss, uniform_loss, uniform_loss2, angle_loss = centerloss(feats, target, )
        # total_loss = torch.add(interclass_loss, intraclass_loss, alpha=0.5)
        total_loss = 5 * intraclass_loss + 2.0 * triplet_loss + 25.0 * uniform_loss2 
        optimizer_center.zero_grad()
        # total_loss.backward(inputs=list(centerloss.parameters()), retain_graph=True)

        optimizer.zero_grad()
        total_loss.backward()
        # feature_loss = intraclass_loss + 0.5*triplet_loss
        # feature_loss.backward()
        optimizer.step()
        optimizer_center.step()
        all_features.append(feats.data.cpu().numpy())
        all_labels.append(target.data.cpu().numpy())
        acc1 = accuracy_l2_nosubcenter(feats, centers, target)
        batch_size = image.shape[0]
        metric_logger.update(loss=total_loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        # print('Intraclass_loss:', intraclass_loss.item(), 'InterClassLoss:', interclass_loss.item())
        epoch_loss += total_loss.item()

    metric_logger.synchronize_between_processes()
    all_features = np.concatenate(all_features, 0)
    all_labels = np.concatenate(all_labels, 0)
    ax = plot_features(all_features, all_labels)
    ax.figure.savefig(os.path.join("figures_Hypersphere_mc_u2_paper", "lambda_25_fig_%d.png" % epoch))
    plt.close("all")
    # feats_tsne = TSNE(n_components=2).fit_transform(all_features[:10000,:], 0)
    # plot_features(feats_tsne, all_labels[:10000,:])
    print(" *Train Acc@1 {top1.global_avg:.3f} ".format(top1=metric_logger.acc1))
    print("Epoch loss = %.2f" % epoch_loss)
    # Debug centers
    if debug_param % 1 == 0:
        print(centerloss.centers)

    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg


def evaluate_majority_voting(model, centerloss, data_loader, device, epoch, args):
    model.eval(), centerloss.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    if args.distributed:
        centers = centerloss.module.centers
    else:
        centers = centerloss.centers
    header = "Test:"
    all_preds_l2, all_preds_l2_norm, all_labels, all_feats, all_dist_euc_cos = (
        [],
        [],
        [],
        [],
        [],
    )
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, args.print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            feats, _ = model(image)
            preds_l2 = get_l2_pred_nosubcenter(feats, centers, target)
            preds_l2_norm = cosine_similarity(feats, centers, target)
            dist_euc_cos = euc_cos(feats, centers, target)
            all_dist_euc_cos.append(dist_euc_cos.cpu())
            all_preds_l2.append(preds_l2.cpu())
            all_preds_l2_norm.append(preds_l2_norm.cpu())
            all_labels.append(target.cpu())
            all_feats.append(feats.cpu())

    preds_l2_norm = torch.cat(all_preds_l2_norm, 0)
    preds_l2 = torch.cat(all_preds_l2, 0)
    preds_euc_cos = torch.cat(all_dist_euc_cos, 0)
    labels = torch.cat(all_labels, 0)
    test_acc_l2 = torch.mul(preds_l2.eq(labels).view(-1).sum(), 100) / len(labels)
    test_acc_l2_norm = torch.mul(preds_l2_norm.eq(labels).view(-1).sum(), 100) / len(labels)
    test_acc_dist_cos = torch.mul(preds_euc_cos.eq(labels).view(-1).sum(), 100) / len(labels)
    print("Test Acc@1_L2 %.3f" % test_acc_l2)
    print("Test Acc@1_COSINE %.3f" % test_acc_l2_norm)
    print("Test Acc@EUC_COS %.3f" % test_acc_dist_cos)
    # all_feats_tsne = TSNE(n_components=2).fit_transform(torch.cat(all_feats, 0))
    # plot_features(all_feats_tsne, labels.data.numpy())
    return test_acc_l2, test_acc_l2_norm


def main(args):
    save_dir = os.path.join(
        "logs",
        "nirvana",
        "%s_%s_lr%.6f_Nirvana_Expand%d_Epoch%d_Seed%d"
        % (
            args.dataset_name,
            args.Network,
            args.lr,
            args.Expand,
            args.epochs,
            args.Seed,
        ),
    )
    utils.mkdir(save_dir)
    with open(os.path.join(save_dir, "commandline_args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    writer = SummaryWriter(log_dir=save_dir)
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.Seed)
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    dataset = CIFAR10_OSR(known=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], batch_size=args.batch_size, use_gpu=True)
    data_loader = dataset.train_loader
    data_loader_val = dataset.test_loader

    args.num_classes = dataset.num_classes
    print("Creating model")

    model = LeNet(args.num_classes)
    # model = resnet.ResNet18(num_classes=args.num_classes)    # model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.to(device)
    # args.feat_dim = model.linear.weight.shape[1]
    # args.feat_dim = model.fc.weight.shape[1]
    print(args)
    centerloss = nirvana_hypersphere(args.num_classes, args.feat_dim, True, True, device, Expand=args.Expand)
    centerloss.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    optimizer_center = torch.optim.SGD(
        centerloss.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler_Center = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_center, T_max=args.epochs)
    model_without_ddp = model
    centerloss_without_ddp = centerloss
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        centerloss = torch.nn.parallel.DistributedDataParallel(centerloss, device_ids=[args.gpu])
        centerloss_without_ddp = centerloss.module

    best_val_acc1, best_val_acc1_lfw, val_acc1, val_acc1_lfw = 0.0, 0.0, 0.0, 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        centerloss_without_ddp.load_state_dict(checkpoint["centerloss"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        best_val_acc1 = checkpoint["best_val_acc1"]
        best_val_acc1_lfw = checkpoint["best_val_acc1_lfw"]

    if args.test_only:
        return evaluate_majority_voting(model, centerloss, data_loader_val, device, epoch=0, args=args)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        acc1, loss = train_one_epoch(
            model,
            centerloss,
            optimizer,
            optimizer_center,
            data_loader,
            device,
            epoch,
            args,
            epoch,
        )
        writer.add_scalar("train/acc1", acc1, epoch)
        writer.add_scalar("train/loss", loss, epoch)
        lr_scheduler.step()
        if epoch % args.eval_freq == 0 and epoch != 0:
            val_acc1, val_acc1_fc = evaluate_majority_voting(
                model, centerloss, data_loader_val, device, epoch, args=args
            )
            is_best = val_acc1 > best_val_acc1
            best_val_acc1 = max(val_acc1, best_val_acc1)
            writer.add_scalar("val/acc1", val_acc1, epoch)
            writer.add_scalar("val/best_acc1", best_val_acc1, epoch)
            print("%.3f Data Val., %.3f Data Best Val." % (val_acc1, best_val_acc1))

            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "centerloss": centerloss_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "val_acc1": val_acc1,
                "val_acc1_fc": val_acc1_fc,
                "best_val_acc1": best_val_acc1,
                "val_acc1_lfw": val_acc1_lfw,
                "best_val_acc1_lfw": best_val_acc1_lfw,
                "args": args,
            }
            # utils.save_on_master(
            #     checkpoint,
            #     os.path.join(save_dir, 'model_{}.pth'.format(epoch)))
            if is_best:
                utils.save_on_master(checkpoint, os.path.join(save_dir, "best_checkpoint.pth"))
                is_best = False

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    writer.close() 


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "-d",
        "--dataset_name",
        default="MNIST",
        metavar="N",
        help="CIFAR10, CIFAR100, car196, MNIST",
    )
    parser.add_argument("-n", "--Network", default="ResNet18", metavar="N", help="ResNet18, ResNet50")
    parser.add_argument(
        "--epochs",
        default=600,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument("--num_classes", default=200, type=int, metavar="N", help="number of classes")
    parser.add_argument("--Expand", default=12, type=int, metavar="N", help="Expand factor of centers")
    parser.add_argument("--Seed", default=0, type=int, metavar="N", help="Seed")
    parser.add_argument(
        "--feat_dim",
        default=2,
        type=int,
        metavar="N",
        help="feature dimension of the model",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=16,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument("--lr", default=0.0005, type=float, help="initial learning rate, 0.1")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-step-size",
        default=50,
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument("--lamda", default=0.5, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq", default=500, type=int, help="print frequency")
    parser.add_argument("--eval-freq", default=1, type=int, help="print frequency")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--test_only", default=False, help="Only Test the model")
    parser.add_argument("--pretrained", default=False, help="True or False")

    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
