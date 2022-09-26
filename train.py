import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import datasets
from utils.metrics import evaluate
from opt import opt
from utils.comm import generate_model
from utils.loss import DeepSupervisionLoss
from utils.metrics import Metrics
import os
import torch.nn.functional as F


def valid(model, valid_dataloader, total_batch):

    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2', 'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    with torch.no_grad():
        bar = tqdm(enumerate(valid_dataloader), total=total_batch)
        for i, data in bar:
            img, gt = data['image'], data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)

            _recall, _specificity, _precision, _F1, _F2, _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision, F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, IoU_bg= _IoU_bg, IoU_mean= _IoU_mean)

    metrics_result = metrics.mean(total_batch)

    return metrics_result


def train():

    # load model
    print('Loading model......')
    model = generate_model(opt)
    print('Load model:', opt.model)

    # load data
    print('Loading data......')
    train_data = getattr(datasets, opt.dataset)(opt.root, opt.train_data_dir, mode='train')
    train_dataloader = DataLoader(train_data, int(opt.batch_size), shuffle=True, num_workers=opt.num_workers)
    valid_data = getattr(datasets, opt.dataset)(opt.root, opt.valid_data_dir, mode='valid')
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    val_total_batch = int(len(valid_data) / 1)

    # load optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.mt, weight_decay=opt.weight_decay)

    lr_lambda = lambda epoch: pow(1.0 - epoch / opt.nEpoch, opt.power)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # train
    print('Start training')
    print('---------------------------------\n')
    
    results = open('./checkpoints/exp' + str(opt.expID) + "/validResults.txt", "a+")
    best_mIoU = 0
    best_idx = 0
    for epoch in range(opt.nEpoch):
        print('------ Epoch', epoch + 1 + 0)
        model.train()
        total_batch = int(len(train_data) / opt.batch_size)
        bar = tqdm(enumerate(train_dataloader), total=total_batch)
        
        for i, data in bar:
            img = data['image']
            gt = data['label']
        
            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            optimizer.zero_grad()
            output = model(img)
            loss = DeepSupervisionLoss(output, gt)
            loss.backward()

            optimizer.step()
            bar.set_postfix_str('loss: %.5s' % loss.item())

        scheduler.step()

        metrics_result = valid(model, valid_dataloader, val_total_batch)

        print("\nValid Result of epoch %d:" % (epoch + 1 + 0), file=results)
        print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'], metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'], metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']), file=results)
        print("\nValid Result of epoch %d:" % (epoch + 1 + 0))
        print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'], metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'], metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))

        if ((epoch + 1 + 0) % opt.ckpt_period == 0): 
            torch.save(model.state_dict(), './checkpoints/exp' + str(opt.expID)+"/ck_{}.pth".format(epoch + 1 + 0))
        
        if metrics_result['IoU_mean'] > best_mIoU:
            best_idx = epoch + 1 + 0
            best_mIoU = metrics_result['IoU_mean']
            torch.save(model.state_dict(), './checkpoints/exp' + str(opt.expID)+"/ck_{}.pth".format(epoch + 1 + 0))
        print("Epoch %d with best mIoU: %.4f" % (best_idx, best_mIoU))
    print("\nEpoch %d with best mIoU: %.4f" % (best_idx, best_mIoU), file=results)

    results.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    
    if opt.mode == 'train':
        print('---PolypSeg Train---')
        train()

    print('Done')
    

