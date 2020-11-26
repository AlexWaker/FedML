import argparse
import logging
import os
import random
import socket
import sys
import time
from os import path

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, DistributedSampler
from torchvision import transforms, datasets

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


from torch.nn.parallel import DistributedDataParallel as DDP

from fedml_api.distributed.fed_transformer.utils import count_parameters, WarmupCosineSchedule, WarmupLinearSchedule, \
    load_from_pickle_file, save_as_pickle_file
from fedml_api.model.cv.transformer.vit.vision_transformer import VisionTransformer, CONFIGS


def init_ddp():
    # use InfiniBand
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'

    # This the global rank: 0, 1, 2, ..., 15
    global_rank = int(os.environ['RANK'])
    print("int(os.environ['RANK']) = %d" % global_rank)

    # This the globak world_size
    world_size = int(os.environ['WORLD_SIZE'])
    print("world_size = %d" % world_size)

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    local_rank = args.local_rank
    print(f"Running basic DDP example on local rank {local_rank}.")
    return local_rank, global_rank


def get_ddp_model(model, local_rank):
    return DDP(model, device_ids=[local_rank], output_device=local_rank)


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "transformer":
        model_type = 'vit-B_16'
        # pretrained on ImageNet (224x224), and fine-tuned on (384x384) high resolution.
        config = CONFIGS[model_type]
        logging.info("Vision Transformer Configuration: " + str(config))
        num_classes = output_dim
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
        # freeze the backbone
        for param in model.transformer.parameters():
            param.requires_grad = False

        model.load_from(np.load(args.pretrained_dir))
        num_params = count_parameters(model)
        logging.info("Vision Transformer Model Size = " + str(num_params))
    return model


def _extract_features(args, model, train_dl, test_dl):
    model.eval()

    directory_train = "./extracted_features/" + args.dataset + "/"
    path_train = directory_train + "train.pkl"
    directory_test = "./extracted_features/" + args.dataset + "/"
    path_test = directory_test + "test.pkl"

    if not os.path.exists(directory_train):
        os.makedirs(directory_train)
    if not os.path.exists(directory_test):
        os.makedirs(directory_test)

    train_data_extracted_features = dict()
    test_data_extracted_features = dict()
    if path.exists(path_train):
        train_data_extracted_features = load_from_pickle_file(path_train)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(train_dl):
                time_start_test_per_batch = time.time()
                x = x.to(device)
                extracted_feature_x, _ = model.transformer(x)
                # train_data_extracted_features[batch_idx] = (extracted_feature_x[:, 0].cpu().detach(), target)
                train_data_extracted_features[batch_idx] = (extracted_feature_x.cpu().detach(), target)
                time_end_test_per_batch = time.time()
                logging.info("train_local feature extraction - time per batch = " + str(
                    time_end_test_per_batch - time_start_test_per_batch))
        save_as_pickle_file(path_train, train_data_extracted_features)

    if path.exists(path_test):
        test_data_extracted_features = load_from_pickle_file(path_test)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_dl):
                time_start_test_per_batch = time.time()
                x = x.to(device)
                extracted_feature_x, _ = model.transformer(x)
                # test_data_extracted_features[batch_idx] = (extracted_feature_x.cpu().detach(), target)
                test_data_extracted_features[batch_idx] = (extracted_feature_x.cpu().detach(), target)
                time_end_test_per_batch = time.time()
                logging.info("test_local feature extraction - time per batch = " + str(
                    time_end_test_per_batch - time_start_test_per_batch))
        save_as_pickle_file(path_test, test_data_extracted_features)

    return train_data_extracted_features, test_data_extracted_features


def _infer(data_extracted_features):
    time_start_test_per_batch = time.time()
    model.eval()

    test_data_extracted_features = data_extracted_features

    test_loss = test_acc = test_total = 0.
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for batch_idx in test_data_extracted_features.keys():
            (x, labels) = test_data_extracted_features[batch_idx]
            x = x.to(device)
            labels = labels.to(device)

            log_probs = model.head(x)
            loss = criterion(log_probs, labels)
            _, predicted = torch.max(log_probs, -1)
            correct = predicted.eq(labels).sum()
            test_acc += correct.item()
            test_loss += loss.item() * labels.size(0)
            test_total += labels.size(0)

    time_end_test_per_batch = time.time()

    # logging.info("time per _infer = " + str(time_end_test_per_batch - time_start_test_per_batch))
    return test_acc, test_total, test_loss


def eval(epoch, train_data_extracted_features, test_data_extracted_features):
    # train data
    train_tot_correct, train_num_sample, train_loss = _infer(train_data_extracted_features)

    # test data
    test_tot_correct, test_num_sample, test_loss = _infer(test_data_extracted_features)

    # test on training dataset
    train_acc = train_tot_correct / train_num_sample
    train_loss = train_loss / train_num_sample

    # test on test dataset
    test_acc = test_tot_correct / test_num_sample
    test_loss = test_loss / test_num_sample

    wandb.log({"Train/Acc": train_acc, "round": epoch})
    wandb.log({"Train/Loss": train_loss, "round": epoch})
    stats = {'training_acc': train_acc, 'training_loss': train_loss}
    logging.info(stats)

    wandb.log({"Test/Acc": test_acc, "round": epoch})
    wandb.log({"Test/Loss": test_loss, "round": epoch})
    stats = {'test_acc': test_acc, 'test_loss': test_loss}
    logging.info(stats)


def train(epoch, epoch_loss, criterion, optimizer, scheduler, train_data_extracted_features):
    # training
    model.train()
    batch_loss = []
    for batch_idx in train_data_extracted_features.keys():
        (x, labels) = train_data_extracted_features[batch_idx]
        x = x.to(device)
        labels = labels.to(device)
        # x, label_a, label_b, lam = mixup_data(x, labels)
        # logging.info(images.shape)
        optimizer.zero_grad()
        log_probs = model.head(x)
        # print(log_probs.shape)
        # print(labels.shape)
        loss = criterion(log_probs, labels)
        # loss = mixup_criterion(criterion, log_probs, label_a, label_b, lam)
        loss.backward()
        # according to ViT paper, all fine-tuned tasks do gradient clipping at global norm 1.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        batch_loss.append(loss.item())
        # if len(batch_loss) > 0:
        #     epoch_loss.append(sum(batch_loss) / len(batch_loss))
        #     logging.info('(Training Epoch: {}\tBatch:{}\tLoss: {:.6f}'.format(epoch, batch_idx,
        #                                                                       sum(epoch_loss) / len(epoch_loss)))


def train_and_eval(model, train_dl, test_dl, args, device):
    criterion = nn.CrossEntropyLoss().to(device)
    if args.client_optimizer == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr,
                                     weight_decay=args.wd, amsgrad=True)

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps,
                                         t_total=args.epochs)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                         t_total=args.epochs)

    train_data_extracted_features, test_data_extracted_features = _extract_features(args, model, train_dl, test_dl)

    epoch_loss = []
    for epoch in range(args.epochs):
        train(epoch, epoch_loss, criterion, optimizer, scheduler, train_data_extracted_features)
        eval(epoch, train_data_extracted_features, test_data_extracted_features)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def load_cifar_centralized_training_for_vit(args):
    if args.is_distributed == 1:
        torch.distributed.barrier()

    """
    the std 0.5 normalization is proposed by BiT (Big Transfer), which can increase the accuracy around 3%
    """
    # CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    # CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    CIFAR_MEAN = [0.5, 0.5, 0.5]
    CIFAR_STD = [0.5, 0.5, 0.5]

    """
    transforms.RandomSizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)) leads to a very low training accuracy.
    """
    transform_train = transforms.Compose([
        # transforms.RandomSizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.Resize(args.img_size),
        transforms.RandomCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=args.data_dir,
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir,
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.is_distributed == 0 else None
    else:
        trainset = datasets.CIFAR100(root=args.data_dir,
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_dir,
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.is_distributed == 0 else None

    if args.is_distributed == 1:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.is_distributed == 0 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DDP Demo")
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument('--model', type=str, default='transformer', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")

    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 0.03)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument("--warmup_steps", default=3, type=int,
                        help="Step of training to perform learning rate warmup for.")

    parser.add_argument('--epochs', type=int, default=30, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")

    parser.add_argument("--pretrained_dir", type=str,
                        default="./../../../fedml_api/model/cv/pretrained/Transformer/vit/ViT-B_16.npz",
                        help="Where to search for pretrained vit models.")

    parser.add_argument("--is_distributed", default=0, type=int,
                        help="Resolution size")
    args = parser.parse_args()
    print(args)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format=' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info("#############process ID = " +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # DDP
    if args.is_distributed == 1:
        local_rank, global_rank = init_ddp()
    else:
        local_rank = args.local_rank
        global_rank = 0

    # GPU
    device = torch.device("cuda:" + str(local_rank))

    # Wandb
    if global_rank == 0:
        wandb.init(
            # project="federated_nas",
            project="fed_transformer",
            name="FedTransformer(c)" + str(args.epochs) + "-lr" + str(args.lr),
            config=args
        )

    # Dataset
    if args.dataset == "cifar10":
        train_dl, test_dl = load_cifar_centralized_training_for_vit(args)
        class_num = 10
    elif args.dataset == "cifar100":
        train_dl, test_dl = load_cifar_centralized_training_for_vit(args)
        class_num = 100
    else:
        train_dl, test_dl = load_cifar_centralized_training_for_vit(args)
        class_num = 10

    # Model
    model = create_model(args, model_name=args.model, output_dim=class_num)
    model.to(device)
    if args.is_distributed == 1:
        model = get_ddp_model(model, local_rank)

    train_and_eval(model, train_dl, test_dl, args, device)

    if args.is_distributed == 1:
        dist.destroy_process_group()
