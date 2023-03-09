import os
import sys
sys.path.append(os.getcwd())
from mtcnn.core.imagedb import ImageDB
import mtcnn.train_net.train as train
import mtcnn.config as config



def train_net(train_file, val_file, model_store_path,
                end_epoch=16, frequent=200, lr=0.01, batch_size=128, use_cuda=True):

    imagedb = ImageDB(train_file)
    gt_imdb = imagedb.load_imdb()

    imagedb_val = ImageDB(val_file)
    gt_val_imdb = imagedb_val.load_imdb()

    # gt_imdb = imagedb.append_flipped_images(gt_imdb)

    train.train_onet(model_store_path=model_store_path, end_epoch=end_epoch, imdb=gt_imdb, imdb_val=gt_val_imdb, batch_size=batch_size, frequent=frequent, base_lr=lr, use_cuda=use_cuda)

if __name__ == '__main__':

    print('train ONet argument:')

    train_file = "/data/VOC/ABBY/darknet/version3.0.1/train.txt"
    val_file = "/data/VOC/ABBY/darknet/version3.0.1/val.txt"
    model_store_path = "/home/ljx/Code/200sever/work/ljx/mtcnn-pytorch/result/"
    end_epoch = 120
    lr = 0.001
    batch_size = 64

    use_cuda = True
    frequent = 50


    train_net(train_file, val_file, model_store_path,
                end_epoch, frequent, lr, batch_size, use_cuda)
