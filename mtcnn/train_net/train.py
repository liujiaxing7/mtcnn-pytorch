import math

from matplotlib import pyplot as plt
from torch import nn
from torch.optim import lr_scheduler

from mtcnn.core.image_reader import TrainImageReader,TestImageLoader
import datetime
import os
from mtcnn.core.models import PNet,RNet,ONet,LossFn
import torch
from torch.autograd import Variable
import mtcnn.core.image_tools as image_tools
import numpy as np
from tensorboardX import SummaryWriter


def compute_accuracy(prob_cls, gt_cls):

    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    #we only need the detection which >= 0
    mask = torch.ge(gt_cls,0)
    #get valid element
    valid_gt_cls = torch.masked_select(gt_cls,mask)
    valid_prob_cls = torch.masked_select(prob_cls,mask)
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls,0.6).float()
    right_ones = torch.eq(prob_ones,valid_gt_cls).float()

    ## if size == 0 meaning that your gt_labels are all negative, landmark or part

    return torch.div(torch.mul(torch.sum(right_ones),float(1.0)),float(size))  ## divided by zero meaning that your gt_labels are all negative, landmark or part


def train_pnet(model_store_path, end_epoch,imdb,
              batch_size,frequent=10,base_lr=0.01,use_cuda=True):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = PNet(is_train=True, use_cuda=use_cuda)
    net.train()

    if use_cuda:
        net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data=TrainImageReader(imdb,12,batch_size,shuffle=True)

    frequent = 10
    for cur_epoch in range(1,end_epoch+1):
        train_data.reset() # shuffle

        for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
            # gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                # gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred = net(im_tensor)
            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            # landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = cls_loss*1.0+box_offset_loss*0.5

            if batch_idx %frequent==0:
                accuracy=compute_accuracy(cls_pred,gt_label)

                show1 = accuracy.data.cpu().numpy()
                show2 = cls_loss.data.cpu().numpy()
                show3 = box_offset_loss.data.cpu().numpy()
                # show4 = landmark_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(),cur_epoch,batch_idx, show1,show2,show3,show5,base_lr))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), os.path.join(model_store_path,"pnet_epoch_%d.pt" % cur_epoch))
        torch.save(net, os.path.join(model_store_path,"pnet_epoch_model_%d.pkl" % cur_epoch))


def train_rnet(model_store_path, end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,use_cuda=True):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = RNet(is_train=True, use_cuda=use_cuda)
    net.train()
    if use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data=TrainImageReader(imdb,24,batch_size,shuffle=True)


    for cur_epoch in range(1,end_epoch+1):
        train_data.reset()

        for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
            gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred = net(im_tensor)
            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            # landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = cls_loss*1.0+box_offset_loss*0.5

            if batch_idx%frequent==0:
                accuracy=compute_accuracy(cls_pred,gt_label)

                show1 = accuracy.data.cpu().numpy()
                show2 = cls_loss.data.cpu().numpy()
                show3 = box_offset_loss.data.cpu().numpy()
                # show4 = landmark_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(), cur_epoch, batch_idx, show1, show2, show3, show5, base_lr))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), os.path.join(model_store_path,"rnet_epoch_%d.pt" % cur_epoch))
        torch.save(net, os.path.join(model_store_path,"rnet_epoch_model_%d.pkl" % cur_epoch))

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def train_onet(model_store_path, end_epoch, imdb, imdb_val,
              batch_size,frequent=50,base_lr=0.01,use_cuda=True):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = ONet(is_train=True)
    net.train()
    print(use_cuda)
    writer = SummaryWriter(model_store_path)
    if use_cuda:
        net.cuda()

    lf = one_cycle(1, 0.2, end_epoch)
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    train_data = TrainImageReader(imdb,48,batch_size,shuffle=True)
    val_data = TrainImageReader(imdb_val,48,batch_size,shuffle=True)

    best_val_loss = 1000
    loss = []
    for cur_epoch in range(1,end_epoch+1):

        train_data.reset()
        val_data.reset()
        loss_epoch=[]
        for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):
            # print("batch id {0}".format(batch_idx))
            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
            gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()

            landmark_offset_pred = net(im_tensor)

            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            # cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            # box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            # landmark_loss = nn.MSELoss()(landmark_offset_pred,gt_landmark)
            landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = landmark_loss
            loss_epoch.append(all_loss.cpu().detach().numpy())


            if batch_idx%frequent==0:
                # accuracy=compute_accuracy(cls_pred,gt_label)

                # show1 = accuracy.data.cpu().numpy()
                # show2 = cls_loss.data.cpu().numpy()
                # show3 = box_offset_loss.data.cpu().numpy()
                show4 = landmark_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()
                lr = [x['lr'] for x in optimizer.param_groups]
                print("%s : Epoch: %d, Step: %d, landmark loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(),cur_epoch,batch_idx, show4,show5,lr[0]))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        scheduler.step()
        lr = [x['lr'] for x in optimizer.param_groups]
        avg_loss_epoch = sum(loss_epoch)/len(loss_epoch)
        loss.append(avg_loss_epoch)

        writer.add_scalar('landmark loss',
                          avg_loss_epoch,
                          cur_epoch )

        writer.add_scalar('lr',
                          lr,
                          cur_epoch )

        landmark_valloss_all = []
        for batch_idx_val, (image_val, (gt_label_val, gt_bbox_val, gt_landmark_val)) in enumerate(val_data):
            im_tensor_val = [ image_tools.convert_image_to_tensor(image_val[i,:,:,:]) for i in range(image_val.shape[0]) ]
            im_tensor_val = torch.stack(im_tensor_val)
            im_tensor_val = Variable(im_tensor_val)
            gt_label_val = Variable(torch.from_numpy(gt_label_val).float())

            gt_bbox_val = Variable(torch.from_numpy(gt_bbox_val).float())
            gt_landmark_val = Variable(torch.from_numpy(gt_landmark_val).float())

            if use_cuda:
                im_tensor_val = im_tensor_val.cuda()
                gt_label_val = gt_label_val.cuda()
                gt_bbox_val = gt_bbox_val.cuda()
                gt_landmark_val = gt_landmark_val.cuda()

            landmark_offset_pred_val = net(im_tensor_val)

            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            # cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            # box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            # landmark_loss = nn.MSELoss()(landmark_offset_pred,gt_landmark)
            landmark_loss_val = lossfn.landmark_loss(gt_label_val,gt_landmark_val,landmark_offset_pred_val)
            landmark_valloss_all.append(landmark_loss_val.cpu().detach().numpy())

        print(landmark_valloss_all)
        landmark_loss_valepcoh = sum(landmark_valloss_all)/len(landmark_valloss_all)
        print("----------------------------------------------------------")
        print("%s : Epoch val: %d, landmark loss: %s" % (datetime.datetime.now(), cur_epoch, landmark_loss_valepcoh))
        print("----------------------------------------------------------")

        if landmark_loss_valepcoh < best_val_loss:
            torch.save(net.state_dict(), os.path.join(model_store_path,"best.pt"))
            best_val_loss = landmark_loss_valepcoh
            # torch.save(net, os.path.join(model_store_path,"onet_epoch_model_%d.pkl" % cur_epoch))


        if cur_epoch % 10 == 0:
            torch.save(net.state_dict(), os.path.join(model_store_path,"onet_epoch_%d.pt" % cur_epoch))
            # torch.save(net, os.path.join(model_store_path,"onet_epoch_model_%d.pkl" % cur_epoch))
    plt.figure()
    plt.plot(loss, 'b', label='landmark_loss')
    plt.ylabel('landmark_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(model_store_path, "landmark_loss.jpg"))