from __future__ import print_function
import os
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import argparse
import dataloader_cifar as dataloader
from function import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--seed', default=42, type=int)
# =============== Noise setting ===========================
parser.add_argument('--noise_mode',  default='sym', help='[sym, pair, instance]')
parser.add_argument('--r', default=0.4, type=float, help='noise ratio')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--num_class', default=100, type=int)

# =============== Training setting ==================
parser.add_argument('--model', default='resnet18', type=str)
parser.add_argument('--feat_len', default=128, type=int)
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--wdecay', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--warm_up', default=10, type=int)
parser.add_argument('--T', default=3, type=int)
parser.add_argument('--threshold', default=0.3, type=float)
parser.add_argument('--penal_coeff', default=0.3, type=float)
parser.add_argument('--ranking', default=0.4, type=float)
parser.add_argument('--lambda_u', default=25, type=int)
parser.add_argument('--run_type', default='ours', type=str, help='["ours", "base", "w_relabel"]')
parser.add_argument('--trade_off', default=0.1, type=float)

parser.add_argument('--log_path', default='', type=str)
parser.add_argument('--gpuid', default=2, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def create_model():
    if args.run_type == 'semi':
        from PreResNet import ResNet18
        model = ResNet18(num_classes=args.num_class)
        model = model.cuda()
        pprint('| Building Pre ResNet-18', test_log)
    else:
        from resnet import ResNet18, ResNet34
        if args.dataset == 'cifar10':
            model = ResNet18(num_classes=args.num_class)
            model = model.cuda()
            pprint('| Building ResNet-18', test_log)
        else:
            model = ResNet34(num_classes=args.num_class)
            model = model.cuda()
            pprint('| Building ResNet-34', test_log)
    return model

test_log = open('./results/%s_%.2f_%s_T(%s)_penCoef(%s)_thres(%s)_ranking(%s)_runType(%s)_base_5e-4'%(
                args.dataset, args.r, args.noise_mode, str(args.T), str(args.penal_coeff),
                str(args.threshold), str(args.ranking), str(args.run_type))+'_stats.txt','w')

if args.dataset=='cifar10':
    args.data_path = './cifar-10'
    args.num_class = 10
    args.warm_up = 10
elif args.dataset=='cifar100':
    args.data_path = './cifar-100'
    args.num_class = 100
    args.warm_up = 30

pprint('| Initialize noisy data', test_log)
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                     num_workers=8, root_dir=args.data_path, log=test_log,
                                     args=args,noise_file='%s/%.1f_%s'%(args.data_path, args.r, args.noise_mode))

net1 = create_model()
cudnn.benchmark = True

if args.r == 0.2:
    args.wdecay = 1e-4
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)

confidence_mb = torch.zeros((50000, args.T))
entropy_list = torch.zeros(50000).cuda()
best_acc = 0.0

pprint('| Start training \n', test_log)
for epoch in range(args.num_epochs+1):   
    lr = args.lr
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    lr = optimizer1.param_groups[0]['lr']

    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch < args.warm_up:
        warmup_trainloader = loader.run('warmup')
        pprint('| Warmup Net1, lr=%.4f' % (lr), test_log)
        warmup(epoch, net1, optimizer1, warmup_trainloader, args, test_log)
    else:
        # training with entropy-based selection
        pprint('| Train Net1, lr=%.4f' % (lr), test_log)
        choice_list = selectiion_phase(confidence_mb, entropy_list, epoch, args)
        label_loader, unlabel_loader = loader.run(mode='train', choice=choice_list)

        train_base(net1, epoch, optimizer1, label_loader, args, test_log)

    test_acc = test(epoch, net1, test_loader, test_log)
    confidence_mb = eval_train(net1, eval_loader, args, epoch, entropy_list, test_log, confidence_MB=confidence_mb)

    if test_acc > best_acc:
        best_acc = test_acc
pprint('best_acc: %.3f' % (best_acc), test_log)
