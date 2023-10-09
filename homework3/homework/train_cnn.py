import sys
from .models import (CNNClassifier, save_model,
                     INPUTNORM1,
                     RANDCROP1, RANDFLIP1, RANDCOLOR1,
                     DROPOUT1, PATIENCE1, BRIGHTNESS1,
                     CONTRAST1, ZRO1, STATAUG1, ADAM1,
                     EPOCHS1, BS1, LR1, START_SIZE_AND_DEPTH1)
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
from torch import optim as O
from numpy import mean
import torch.utils.tensorboard as tb

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

def train(args):
    from os import path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # stats_data = load_data(path.join('data', 'train'), batch_size=1, augment=STATAUG1)
    model = CNNClassifier(n_input_channels=3, dropout=DROPOUT1)
    model.to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    from time import time
    t_0 = time()

    loss = torch.nn.CrossEntropyLoss().to(device)
    if ADAM1:
        optimizer = O.Adam(model.parameters(), lr=LR1, weight_decay=args.wd)
    else:
        optimizer = O.SGD(model.parameters(), lr=LR1, momentum=args.mom, weight_decay=args.wd)

    train_data = load_data(path.join('data', 'train'), batch_size=BS1, augment=STATAUG1)
    valid_data = load_data(path.join('data', 'valid'), batch_size=BS1)
    global_step = 0
    if PATIENCE1 != 0:
        scheduler = O.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=PATIENCE1, factor=0.5)

    for epoch in range(EPOCHS1):
        model.train()
        train_accuracy = []
        valid_accuracy = []
        for batch_data, batch_label in train_data:
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)
            o = model(batch_data)
            loss_val = loss(o, batch_label)

            train_logger.add_scalar(path.join('loss'), loss_val, global_step=global_step)
            train_accuracy.append(accuracy(o, batch_label).cpu().detach())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1
        model.eval()
        for vbatch_data, vbatch_label in valid_data:
            vbatch_data, vbatch_label = vbatch_data.to(device), vbatch_label.to(device)
            o = model(vbatch_data)
            valid_accuracy.append(accuracy(o, vbatch_label).cpu().detach())
        train_logger.add_scalar(path.join('accuracy'), mean(train_accuracy), global_step=global_step)
        valid_logger.add_scalar(path.join('accuracy'), mean(valid_accuracy), global_step=global_step)
        if PATIENCE1 != 0:
            scheduler.step(mean(valid_accuracy))
            print('Epoch: ' + str(epoch) + ' | ' + 'LR: ' + str(optimizer.param_groups[0]['lr']))

    mins = str(int((time() - t_0)//60))
    seconds = round((((time() - t_0)/60) % 1)*60)
    if seconds > 10:
        seconds = str(seconds)
    else:
        seconds = str(0) + str(seconds)

    LAYRS = [START_SIZE_AND_DEPTH1[0]]
    tmp = START_SIZE_AND_DEPTH1[0]
    for lyr in range(START_SIZE_AND_DEPTH1[1]):
        tmp *= 2
        LAYRS.append(tmp)

    print('Epochs: ' + str(EPOCHS1) + ' | ' +
          'BS: ' + str(BS1) + ' | ' +
          'LR: ' + str(LR1) + ' | ' +
          'Lyrs: ' + str(LAYRS) + ' | ' +
          'Pat: ' + str(PATIENCE1) + ' | ' +
          'InNorm: ' + str(INPUTNORM1) + ' | ' +
          'RCrop: ' + str(RANDCROP1) + ' | ' +
          'RFlip: ' + str(RANDFLIP1) + ' | ' +
          'RColor: ' + str(RANDCOLOR1) + ' | ' +
          'Bright: ' + str(BRIGHTNESS1) + ' | ' +
          'Contrast: ' + str(CONTRAST1) + ' | ' +
          'Z: ' + str(ZRO1) + ' | ' +
          'ADAM: ' + str(ADAM1) + ' | ' +
          'StatAug: ' + str(STATAUG1)
          )
    print('Train time: ' + mins + ":" + seconds + ' | ' + 'Train Accuracy: ' + str(mean(train_accuracy)) + ' | ' + 'Valid Accuracy: ' + str(mean(valid_accuracy)))
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='log')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-3)
    # parser.add_argument('--layers', nargs='+', type=int, default=[32, 64, 128])

    args = parser.parse_args()
    train(args)

