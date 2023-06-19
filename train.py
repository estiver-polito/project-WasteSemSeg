#%%
import os
import random
import copy
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from model import ENet
from icnet import ICNet
from bisenet_v2 import BiSeNet
from config import cfg
from loading_data import loading_data
from utils import *
from timer import Timer
from matplotlib import pyplot as plt
from loss import *
from ptflops import get_model_complexity_info

from score import SegmentationMetric
import matplotlib.image as mpimg
import pdb

exp_name = cfg.TRAIN.EXP_NAME
device = ""
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()
metric = SegmentationMetric(cfg.DATA.NUM_CLASSES)
class_names = ['none','paper', 'bottle', 'alluminium', 'Nylon']
best_results =  {
  "none": 0.0,
  "paper": 0.0,
  "bottle": 0.0,
  "alluminium": 0.0,
  "Nylon": 0.0,
  "total": 0.0,
  "model":[]
}


def check_image(image,mask):
    #inputs, labels = data
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    #restore_transform(inputs[0])
    plt.imshow(restore_transform(image), cmap='gray')
    plt.subplot(122)
    plt.imshow(mask, cmap='gray')
    plt.show()
    
    

def main():

    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    # if len(cfg.TRAIN.GPU_ID)==1:
    #     torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    # torch.backends.cudnn.benchmark = True

    net = []   
    #net = ICNet(1)
    #et = ENet(only_encode=True)
    #net = net.to("cpu")
    if cfg.TRAIN.STAGE=='all':
        if cfg.MODEL.MODEL == "icnet":
            net = ICNet(cfg.DATA.NUM_CLASSES)
        elif cfg.MODEL.MODEL == "enet":    
            net = ENet(only_encode=False)
        elif cfg.MODEL.MODEL == "bisenet":
            net = BiSeNet(cfg.DATA.NUM_CLASSES,True)
        if cfg.TRAIN.PRETRAINED_ENCODER != '':
            encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
            del encoder_weight['classifier.bias']
            del encoder_weight['classifier.weight']
            # pdb.set_trace()
            net.encoder.load_state_dict(encoder_weight)
    elif cfg.TRAIN.STAGE =='encoder':
        net = ENet(only_encode=True)


    if len(cfg.TRAIN.GPU_ID)>1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net=net.to(device)
   
    net.train()

    if cfg.MODEL.MODEL == "enet":
        criterion =  torch.nn.CrossEntropyLoss().to(device) if cfg.DATA.NUM_CLASSES > 1 else torch.nn.BCEWithLogitsLoss().to(device)
    elif cfg.MODEL.MODEL == "icnet":
        criterion =  Multiclass_ICNetLoss().to(device)  if cfg.DATA.NUM_CLASSES > 1 else Binary_ICNetLoss().to(device)
    elif cfg.MODEL.MODEL == "bisenet":
        criterion = MixSoftmaxCrossEntropyLoss(True).to(device)
    

    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    _t = {'train time' : Timer(),'val time' : Timer()} 
    validate(val_loader, net, criterion, optimizer, -1, restore_transform)
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))
        _t['val time'].tic()
        validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        _t['val time'].toc(average=False)
        print('val time of  epoch {}: {:.2f}s'.format(epoch,_t['val time'].diff))

    benchmark(next(iter(val_loader))[0].shape)
    

def benchmark(dsize):

    total_flops, _ = get_model_complexity_info(best_results['model'], (3, 224, 448), as_strings=True,
                                        print_per_layer_stat=False, verbose=False,flops_units="GMac")
   
    
    param_size = 0
    for param in best_results['model'].parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in best_results['model'].buffers():
        buffer_size += buffer.nelement() * buffer.element_size()


    print("Best Model ")
    if cfg.DATA.NUM_CLASSES > 1:
        for v in list(best_results.keys())[:-1]:
            print(f'Mean IoU for {v}: {best_results[v]:.6f}')
    
    print('[mean iu %.4f]' % (best_results['total']))
    print("Performance {} GFlops".format(float(total_flops.split(" ")[0]) * 2))
    print("Number Parameters {}".format(param_size))
    print("Size Model {} MB".format((param_size + buffer_size) / 1024**2))


    

def train(train_loader, net, criterion, optimizer, epoch):
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # inputs = Variable(inputs).cuda()
        # labels = Variable(labels).cuda()
        inputs = inputs.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()
        outputs = net(inputs)
        if cfg.MODEL.MODEL == "enet" and cfg.DATA.NUM_CLASSES == 1 :
            loss = criterion(outputs, labels.unsqueeze(1).float())
        else:
            loss = criterion(outputs, labels)
   
        # optimizer.zero_grad()
        # outputs = net(inputs)
        # _, pred_sub4, pred_sub8, pred_sub16 = net(inputs)
        # labels = labels.unsqueeze(1).float()
        # target_sub4 = F.interpolate(labels, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        # target_sub8 = F.interpolate(labels, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        # target_sub16 = F.interpolate(labels, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        # loss = criterion(pred_sub4, target_sub4)
        # loss += criterion(pred_sub8, target_sub8)
        # loss += criterion(pred_sub16, target_sub16)
        
        loss.backward()
        optimizer.step()
        print(f"{i+1}/{len(train_loader)}")
        
    


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    iou_ = 0.0
    
    iou_sum_classes = [0.0] * cfg.DATA.NUM_CLASSES
    metric.reset()
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)
       
        # inputs = Variable(inputs, volatile=True).cuda()
        # labels = Variable(labels, volatile=True).cuda()
        outputs = net(inputs)
        
        if not cfg.MODEL.MODEL == "enet":
            outputs = outputs[0]
        
        #metric.update(outputs, labels)
        #outputs,_,_,_ = net(inputs)
        #for binary classification
        if cfg.DATA.NUM_CLASSES == 1:
            outputs[outputs>0.5] = 1
            outputs[outputs<=0.5] = 0
            x , _ = calculate_mean_iu(outputs.squeeze_(1).data.cpu().numpy(), labels.data.cpu().numpy(), 2)
            iou_ += x
        else:
            x , y = calculate_mean_iu(outputs.argmax(dim=1).data.cpu().numpy(), labels.data.cpu().numpy(), cfg.DATA.NUM_CLASSES)
            iou_ += x
            iou_sum_classes = [sum(x) for x in zip(iou_sum_classes, y)]

            
             
            
            # for c in range(cfg.DATA.NUM_CLASSES):
            # # predmask
            #     pred_mask = (outputs.argmax(dim=1) == c).cpu().numpy()
            #     labels_mask = (labels == c).cpu().numpy()
            #     class_iou = calculate_mean_iu(pred_mask, labels_mask, 2)
            #     iou_sum_classes[c] += class_iou



    #IoU,mIoU = metric.get()
    if cfg.DATA.NUM_CLASSES == 1:
        
        print('[mean iu %.4f]' % (iou_ / len(val_loader)))

        if iou_/len(val_loader) > best_results["total"]:
            
            best_results["total"] = iou_/len(val_loader)
            best_results["model"] = net

        
        
    else:
        #mean_iu_classes = [x for x in IoU]
        mean_iu_classes = [x / len(val_loader) for x in iou_sum_classes]
        # Print the mean IoU for each class
        class_names = ['none','paper', 'bottle', 'alluminium', 'Nylon']
        for i, class_name in enumerate(class_names):
            #print(f'Mean IoU for {class_name}: {IoU[i]}')
            print(f'Mean IoU for {class_name}: {mean_iu_classes[i]:.6f}')
        print('[mean iu %.4f]' % (iou_/len(val_loader) ))
    
        if iou_/len(val_loader) > best_results["total"]:
            best_iou = iou_/len(val_loader)
        
            for i,v in enumerate(list(best_results.keys())[:-2]):
                best_results[v] = mean_iu_classes[i]

            best_results["total"] = iou_/len(val_loader)
            best_results["model"] = net

            # flops, params = get_model_complexity_info(net, (3, 224, 448), as_strings=True,
            #                             print_per_layer_stat=False, verbose=False)
   
    net.train()
    criterion.cuda()    


if __name__ == '__main__':
    if cfg.TRAIN.CUDA and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    main()



# %%
