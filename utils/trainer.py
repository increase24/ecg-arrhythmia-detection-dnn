import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from .stat import AverageMeter, accuracy
from .saver import plot_confusion_matrix


def train_epoch(train_loader, model, device, criterion, optimizer, epoch, print_freq):
    losses = AverageMeter()
    accuracies = AverageMeter()
    model.train()
    for idx, (input, target) in enumerate(train_loader):
        # compute output and loss
        input, target = input.to(device), target.to(device)
        model.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))
        # measure accuracy
        [acc] = accuracy(output.detach(), target.detach().cpu())
        accuracies.update(acc.item(), input.size(0))
        # compute grandient and do back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                    epoch, idx, len(train_loader),
                    loss = losses, acc = accuracies 
                ))
    print('Epoch: [{0}][{1}/{2}]\t'
            'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                epoch, idx, len(train_loader),
                loss = losses, acc = accuracies 
            ))
    return losses.avg, accuracies.avg
            
            
def validate(valid_loader, model, device, criterion, print_freq, eval_only):
    losses = AverageMeter()
    accuracies = AverageMeter()
    logits_matrix = []
    targets_list = []
    model.eval()
    with torch.no_grad():
        for idx, (input, target) in enumerate(valid_loader):
            input, target = input.to(device), target.to(device)
            # compute output and loss
            output = model(input)
            loss = criterion(output, target)
            if eval_only:
                logits_matrix.append(output.detach().cpu().argmax(dim=1).numpy())
                targets_list.append(target.detach().cpu().numpy())
            #measure accuracy and record loss
            [acc] = accuracy(output.detach(), target.detach().cpu())
            losses.update(loss.item(), input.size(0))
            accuracies.update(acc.item(), input.size(0))
            if idx % print_freq == 0:
                print(
                    'Test: [{0}/{1}]\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                    idx, len(valid_loader),
                    loss = losses, acc = accuracies 
                ))
        print('Test: [{0}/{1}]\t'
            'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
            idx, len(valid_loader),
            loss = losses, acc = accuracies 
        ))
        if eval_only:
            logits_matrix = np.concatenate(logits_matrix)
            targets_list = np.concatenate(targets_list)
            print(classification_report(targets_list, logits_matrix, target_names = ['A', 'N', 'O', '~'], digits=3))
            cm = confusion_matrix(logits_matrix, targets_list)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plot_confusion_matrix(cm_normalized, ['A', 'N', 'O', '~'], './figs/cm.png')
    return losses.avg, accuracies.avg


