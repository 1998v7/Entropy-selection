import sys, torch
import torch.nn.functional as F
import numpy as np
import random


def one_lossF(output, one_hot):
    log_prob = torch.nn.functional.log_softmax(output, dim=1)
    loss = - torch.sum(log_prob * one_hot) / output.size(0)
    return loss


def warmup(epoch,net,optimizer,dataloader, args, log):
    net.train()
    correct = 0
    total = 0
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs, _= net(inputs)

        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += predicted.eq(labels).cpu().sum().item()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()


    train_acc = 100. * correct / total
    pprint('\n%s:%.1f-%s | Epoch [%3d/%3d]  train_acc: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, train_acc), log)


def test(epoch, net1, test_loader, test_log):
    net1.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, _ = net1(inputs)
            _, predicted = torch.max(outputs1, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    pprint("| Test Epoch #%d\t Test Accuracy: %.2f%%" % (epoch, acc), test_log)
    return acc


def pop(confidence_mb, value=None, index=None, args=None, type='warm_up'):
    for i in range(args.T-1):
        confidence_mb[index, i] = confidence_mb[index, i+1]
    confidence_mb[index, args.T-1] = value
    return confidence_mb


def eval_train(model1, eval_loader, args, epoch, entropy_list, log, confidence_MB=None):
    model1.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, _ = model1(inputs)

            confidence = torch.softmax(outputs1, dim=-1)
            probability, predicted1 = torch.max(confidence, 1)

            for i in range(inputs.size(0)):
                if predicted1[i] == targets[i]:
                    pop(confidence_MB, value=probability[i], index=index[i], args=args)
                else:
                    pop(confidence_MB, value=-1, index=index[i], args=args)

            _, predicted = torch.max(outputs1, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
        acc = 100. * correct / total
        pprint("| Train Accuracy: %.2f%%\n" % (acc), log)
    # evalidate
    return confidence_MB


def selectiion_phase(confidence_MB, entropy_list, epoch, args):
    for i in range(len(confidence_MB)):
        vector = confidence_MB[i]  # [1, args.T]
        # penalization
        vector_nozero = 1.0 + vector
        error_size = vector.size()[0] - torch.nonzero(vector_nozero).size()[0]
        penality = args.penal_coeff * error_size
        # softmax
        vector = torch.softmax(vector.unsqueeze(dim=0), dim=1)
        entropy = -torch.sum(vector.log() * vector, dim=1) - penality
        entropy_list[i] = entropy

    entropy_list = (entropy_list - entropy_list.min()) / (entropy_list.max() - entropy_list.min())
    return (entropy_list.cpu().numpy() > args.threshold)




def train_base(net, epoch, optimizer, label_loader, args, test_log):
    net.train()

    num_iter = (len(label_loader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, label_x, index_x) in enumerate(label_loader):
        inputs_x, inputs_x2, label_x = inputs_x.cuda(), inputs_x2.cuda(), label_x.cuda()
        output_x, _ = net(inputs_x)
        loss = F.cross_entropy(output_x, label_x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t train_loss: %.4f ' %
                             (args.dataset, args.r, args.noise_mode, epoch + 1, args.num_epochs, batch_idx + 1, num_iter, loss))
            sys.stdout.flush()



def pprint(context, log_name):
    print(context)
    log_name.write(context + '\n')
    log_name.flush()