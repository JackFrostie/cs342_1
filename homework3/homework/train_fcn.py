import torch
import numpy as np

from .models import (FCN, save_model,
                     INPUTNORM2, START_SIZE,
                     RANDCROP2, RANDFLIP2, RANDCOLOR2,
                     DROPOUT2, PATIENCE2, BRIGHTNESS2,
                     CONTRAST2, ZRO2, STATAUG2, ADAM2,
                     EPOCHS2, BS2, LR2, DEPTH, DROP_P)
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
from torch import optim as O
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = FCN(dropout=DROPOUT2)
    model.to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    from time import time
    t_0 = time()
    rescale = torch.tensor([0.2/DENSE_CLASS_DISTRIBUTION[0],
                            0.2/DENSE_CLASS_DISTRIBUTION[1],
                            0.2/DENSE_CLASS_DISTRIBUTION[2],
                            0.2/DENSE_CLASS_DISTRIBUTION[3],
                            0.2/DENSE_CLASS_DISTRIBUTION[4]])
    rescale.to(device)

    loss = torch.nn.CrossEntropyLoss(
        reduction='mean',
        weight=rescale
    ).to(device)
    if ADAM2:
        optimizer = O.Adam(model.parameters(), lr=LR2, weight_decay=1e-4)
    else:
        optimizer = O.SGD(model.parameters(), lr=LR2, momentum=0.9, weight_decay=1e-4)

    t_list = []
    if RANDFLIP2:
        t_list.append(dense_transforms.RandomHorizontalFlip())
    if RANDCOLOR2:
        t_list.append(dense_transforms.ColorJitter(brightness=2))
    t_list.append(dense_transforms.ToTensor())

    train_data = load_dense_data(path.join('dense_data', 'train'), batch_size=BS2, transform=dense_transforms.Compose(t_list))
    valid_data = load_dense_data(path.join('dense_data', 'valid'), batch_size=BS2)
    global_step = 0
    if PATIENCE2 != 0:
        scheduler = O.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=PATIENCE2, factor=0.5)

    for epoch in range(EPOCHS2):
        model.train()
        train_class_iou = []
        train_iou = []
        train_global_accuracy = []
        train_class_accuracy = []
        train_average_accuracy = []
        train_per_class = []
        valid_class_iou = []
        valid_iou = []
        valid_global_accuracy = []
        valid_class_accuracy = []
        valid_average_accuracy = []
        valid_per_class = []

        train_tracker = ConfusionMatrix()
        valid_tracker = ConfusionMatrix()
        for batch_data, batch_label in train_data:
            batch_data, batch_label = batch_data.to(device), batch_label.type(torch.LongTensor).to(device)
            o = model(batch_data)
            loss_val = loss(o, batch_label)

            train_logger.add_scalar(path.join('loss'), loss_val, global_step=global_step)
            train_tracker.add(o.argmax(1), batch_label)

            train_class_iou.append(train_tracker.iou.cpu().detach())
            train_iou.append(train_tracker.class_iou.cpu().detach())
            train_global_accuracy.append(train_tracker.global_accuracy.cpu().detach())
            train_class_accuracy.append(train_tracker.class_accuracy.cpu().detach())
            train_average_accuracy.append(train_tracker.average_accuracy.cpu().detach())
            train_per_class.append(train_tracker.per_class.cpu().detach())


            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1
        model.eval()
        for vbatch_data, vbatch_label in valid_data:
            vbatch_data, vbatch_label = vbatch_data.to(device), vbatch_label.to(device)
            o = model(vbatch_data)
            valid_tracker.add(o.argmax(1), vbatch_label)

            valid_class_iou.append(valid_tracker.iou.cpu().detach())
            valid_iou.append(valid_tracker.class_iou.cpu().detach())
            valid_global_accuracy.append(valid_tracker.global_accuracy.cpu().detach())
            valid_class_accuracy.append(valid_tracker.class_accuracy.cpu().detach())
            valid_average_accuracy.append(valid_tracker.average_accuracy.cpu().detach())
            valid_per_class.append(valid_tracker.per_class.cpu().cpu().detach())
            log(valid_logger, vbatch_data, vbatch_label, o, global_step)

        train_logger.add_scalar(path.join('iou'), np.mean(train_class_iou), global_step=global_step)
        train_logger.add_scalar(path.join('class_iou'), np.mean(train_iou), global_step=global_step)
        train_logger.add_scalar(path.join('global_accuracy'), np.mean(train_global_accuracy), global_step=global_step)
        train_logger.add_scalar(path.join('class_accuracy'), np.mean(train_class_accuracy), global_step=global_step)
        train_logger.add_scalar(path.join('average_accuracy'), np.mean(train_average_accuracy), global_step=global_step)
        train_logger.add_scalar(path.join('per_class'), np.mean(train_per_class), global_step=global_step)

        valid_logger.add_scalar(path.join('iou'), np.mean(valid_class_iou), global_step=global_step)
        valid_logger.add_scalar(path.join('class_iou'), np.mean(valid_iou), global_step=global_step)
        valid_logger.add_scalar(path.join('global_accuracy'), np.mean(valid_global_accuracy), global_step=global_step)
        valid_logger.add_scalar(path.join('class_accuracy'), np.mean(valid_class_accuracy), global_step=global_step)
        valid_logger.add_scalar(path.join('average_accuracy'), np.mean(valid_average_accuracy), global_step=global_step)
        valid_logger.add_scalar(path.join('per_class'), np.mean(valid_per_class), global_step=global_step)

        if PATIENCE2 != 0:
            scheduler.step(np.mean(valid_iou))
            print('Epoch: ' + str(epoch) + ' | ' + 'LR: ' + str(optimizer.param_groups[0]['lr']))

    mins = str(int((time() - t_0)//60))
    seconds = round((((time() - t_0)/60) % 1)*60)
    if seconds > 10:
        seconds = str(seconds)
    else:
        seconds = str(0) + str(seconds)

    print('Epochs: ' + str(EPOCHS2) + ' | ' +
          'BS: ' + str(BS2) + ' | ' +
          'LR: ' + str(LR2) + ' | ' +
          'Depth/Size: ' + str([DEPTH, START_SIZE]) + ' | ' +
          'Pat: ' + str(PATIENCE2) + ' | ' +
          'InNorm: ' + str(INPUTNORM2) + ' | ' +
          'RCrop: ' + str(RANDCROP2) + ' | ' +
          'RFlip: ' + str(RANDFLIP2) + ' | ' +
          'RColor: ' + str(RANDCOLOR2) + ' | ' +
          'Bright: ' + str(BRIGHTNESS2) + ' | ' +
          'Contrast: ' + str(CONTRAST2) + ' | ' +
          'Z: ' + str(ZRO2) + ' | ' +
          'ADAM: ' + str(ADAM2) + ' | ' +
          'Drop P: ' + str(DROP_P) + ' | ' +
          'StatAug: ' + str(STATAUG2)
          )

    print('Train time: ' + mins + ":" + seconds + ' | ' +
          'Train IoU: ' + str(np.mean(train_iou)) + ' | ' +
          'Valid IoU: ' + str(np.mean(valid_iou)))

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
