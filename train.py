#%%
import os
import sys
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
import neptune
from neptune_pytorch import NeptuneLogger

from score import SegmentationMetric
import matplotlib.image as mpimg
import pdb
from neptune.utils import stringify_unsupported



exp_name = cfg.TRAIN.EXP_NAME
device = ""
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)
attr_map = dict(item.strip().split('=', 1) for item in sys.argv[1:])
cfg.TRAIN.MAX_EPOCH = int(attr_map['epochs'])
cfg.CONFIG.MODEL = attr_map['model']
cfg.DATA.NUM_CLASSES = int(attr_map['classes'])
cfg.TRAIN.BATCH_SIZE = cfg.VAL.BATCH_SIZE = int(attr_map['batch'])

run = neptune.init_run(
    project="stiver/waste-segmentation",  # replace with your own (see instructions below)
    api_token=attr_map['token'],
)





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
  "total": 0.0
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

    
    cfg_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.py'),"r")  
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
        if cfg.CONFIG.MODEL == "icnet":
            net = ICNet(cfg.DATA.NUM_CLASSES)
        elif cfg.CONFIG.MODEL == "enet":    
            net = ENet(only_encode=False)
        elif cfg.CONFIG.MODEL == "bisenet":
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

    global npt_logger
    npt_logger = NeptuneLogger(run, model=net, log_model_diagram=True, log_parameters=True)
    run[npt_logger.base_namespace]["hyperparams"] = stringify_unsupported(cfg)

    
    net.train()

    if cfg.CONFIG.MODEL == "enet":
        criterion =  torch.nn.CrossEntropyLoss().to(device) if cfg.DATA.NUM_CLASSES > 1 else torch.nn.BCEWithLogitsLoss().to(device)
    elif cfg.CONFIG.MODEL == "icnet":
        criterion =  Multiclass_ICNetLoss().to(device)  if cfg.DATA.NUM_CLASSES > 1 else Binary_ICNetLoss().to(device)
    elif cfg.CONFIG.MODEL == "bisenet":
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

    benchmark()
    

def benchmark():
    

    total_flops, _ = get_model_complexity_info(npt_logger.model, (3, 224, 448), as_strings=True,
                                        print_per_layer_stat=False, verbose=False,flops_units="GMac")
   
    
    param_size = 0
    for param in npt_logger.model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in npt_logger.model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()


    if cfg.DATA.NUM_CLASSES > 1:
        for v in list(best_results.keys())[:-1]:
            print(f'Mean IoU for {v}: {best_results[v]:.6f}')

    features ={
        "performance":"{} GFlops".format(float(total_flops.split(" ")[0]) * 2),
        "Number_Parameters": param_size,
        "Size_Model": "{} MB".format((param_size + buffer_size) / 1024**2)
    }

    run[npt_logger.base_namespace]["best-results"] = stringify_unsupported(best_results)
    run[npt_logger.base_namespace]["model-features"] = stringify_unsupported(features)
    

    run.stop
    # print('[mean iu %.4f]' % (best_results['total']))
    # print("Performance {} GFlops".format(float(total_flops.split(" ")[0]) * 2))
    # print("Number Parameters {}".format(param_size))
    # print("Size Model {} MB".format((param_size + buffer_size) / 1024**2))


    

def train(train_loader, net, criterion, optimizer, epoch):
    mean_loss=0.0
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # inputs = Variable(inputs).cuda()
        # labels = Variable(labels).cuda()
        inputs = inputs.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()
        outputs = net(inputs)
        
        if cfg.CONFIG.MODEL == "enet" and cfg.DATA.NUM_CLASSES == 1 :
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
   
        mean_loss += loss
        
        loss.backward()
        optimizer.step()
        #print(f"{i+1}/{len(train_loader)}")

    run[npt_logger.base_namespace]["train/epoch/loss"].append(mean_loss/len(train_loader))
        
    


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    valid_losses = []
    iou_ = 0.0
    
    iou_sum_classes = [0.0] * cfg.DATA.NUM_CLASSES
    metric.reset()
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)
       
        
        outputs = net(inputs)

        with torch.no_grad():
            if cfg.CONFIG.MODEL == "enet" and cfg.DATA.NUM_CLASSES == 1 :
                los = criterion(outputs, labels.unsqueeze(1).float())
            else:
                los = criterion(outputs, labels)
            
            valid_losses.append(los.item())
        
        if not cfg.CONFIG.MODEL == "enet":
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

    #IoU,mIoU = metric.get()
    if cfg.DATA.NUM_CLASSES == 1:
        
        print('[mean iu %.4f]' % (iou_ / len(val_loader)))

        if iou_/len(val_loader) > best_results["total"]:
            
            best_results["total"] = iou_/len(val_loader)
            best_model = net
            
            npt_logger.save_model("model")

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

        
            for i,v in enumerate(list(best_results.keys())[:-1]):
                best_results[v] = mean_iu_classes[i]

            best_results["total"] = iou_/len(val_loader)
            
            npt_logger.save_model("model")
            

    
    if epoch > 0:
        run[npt_logger.base_namespace]["validation/epoch/mean_iou"].append(iou_/len(val_loader))
        run[npt_logger.base_namespace]["validation/epoch/loss"].append(np.mean(valid_losses))
   
        if cfg.DATA.NUM_CLASSES > 1:
            for i,v in enumerate(list(best_results.keys())[:-1]):
                run[npt_logger.base_namespace]["validation/epoch/{}_iou".format(v)].append(mean_iu_classes[i])
                best_results[v] = mean_iu_classes[i]

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
