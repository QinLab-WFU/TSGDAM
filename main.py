import cv2
import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from options import MyOptions
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from model import MyModel
import math
import time
from skimage.feature import hog
import logging

# import visdom
opt = MyOptions().parse()


# 初始化visdom服务器
# vis = visdom.Visdom()

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_flist = sorted(os.listdir(self.opt.data_root))
        self.mask_flist = sorted(os.listdir(self.opt.mask_root))

        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        fname = self.img_flist[index]
        I_i = cv2.imread(self.opt.data_root + fname)
        I_g = copy.deepcopy(I_i)
        L_i = self.get_feature(copy.deepcopy(I_i))
        # L_i = copy.deepcopy(I_i)
        L_g = copy.deepcopy(L_i)

        I_i = self.transform(I_i)
        I_g = self.transform(I_g)
        L_i = self.transform(L_i)
        L_i = L_i[0, :, :].view(1, 256, 256)
        L_g = self.transform(L_g)
        L_g = L_g[0, :, :].view(1, 256, 256)

        mask = Image.open(self.opt.mask_root + self.mask_flist[index])
        mask = transforms.ToTensor()(mask)
        return {'I_i': I_i, 'I_g': I_g, 'M': mask, 'L_i': L_i, 'L_g': L_g, 'fname': fname}

    def __len__(self):
        return len(self.img_flist)

    def build_filters(self,ksize, sigma, theta, lambd, gamma):
        filters = []
        kmax = math.pi / 2
        for phi in np.arange(0, np.pi, np.pi / 8):
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta + phi, lambd, gamma, ktype=cv2.CV_32F)
            if kern.sum() != 0:
                kern /= 0.1 * kern.sum()
            filters.append(kern)
        return filters

    def process(self,img, filters):
        accum = np.zeros((256, 256, 3), np.uint8)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            fimg = np.dstack((fimg, fimg, fimg))
            np.maximum(accum, fimg, accum)
        return accum

    def get_feature(self,img, ksize=3, sigma=3, theta=0, lambd=100, gamma=0.5):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filters = self.build_filters(ksize, sigma, theta, lambd, gamma)
        res = self.process(img_gray, filters)
        edges = cv2.Canny(img_gray, threshold1=30, threshold2=100)
        edges = cv2.bitwise_not(edges)
        combined_feature = cv2.bitwise_and(res, res, mask=edges)
        return combined_feature

class MyDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = MyDataset(opt)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            num_workers= 8)

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data

def postprocess(img):
    img = img.detach().to('cpu')
    img = img * 127.5 + 127.5
    img = img.permute(0, 2, 3, 1)
    return img.int()


# 计算指标
def metrics(real, fake):
    real = postprocess(real)
    fake = postprocess(fake)
    a = real.numpy()
    b = fake.numpy()
    # ssim = []
    psnr = []
    for i in range(len(a)):
        # ssim.append(compare_ssim(a[i], b[i], win_size=11, data_range=255.0, multichannel=True))
        psnr.append(compare_psnr(a[i], b[i], data_range=255))
    # return np.mean(ssim), np.mean(psnr), m
    return np.mean(psnr)

def print_network_params(net, name):
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))


def train():
    opt.device = 'cuda:0'
    log_file = 'drop'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    with open('1029.txt', 'w') as file:
        for key, value in vars(opt).items():
            file.write(f'{key}: {value}\n')
    opt.data_root = '/home/li/image_inpaint/data/celeba-hq/train/'  # The location of your training data
    opt.mask_root = '/home/li/image_inpaint/data/celeba-hq/cele_ran60/'  # The location of your training data mask
    train_set = MyDataLoader(opt)
    model = MyModel()
    model.initialize(opt)
    print('---------- Networks Parameters -------------')
    for name in ['LBP', 'G']:
        net = getattr(model, 'net' + name)
        print_network_params(net, name)
    print('-------------------------------------------')
    print('Train %d' % (len(train_set)))
    start_time = time.time()
    for epoch in range(1, 30):
        epoch_start_time = time.time()
        print('Epoch: %d' % epoch)
        epoch_iter = 0
        losses_G, psnr, = [], []
        for i, data in enumerate(train_set):
            iter_start_time = time.time()
            fname = data['fname'][0]
            epoch_iter += opt.batchSize
            model.set_input(data)
            I_g, I_o, loss_G, L_o = model.optimize_parameters()
            p = metrics(I_g, I_o)
            psnr.append(p)
            losses_G.append(loss_G.detach().item())
            current_loss_G = losses_G[-1]
            current_psnr = psnr[-1]
            avg_loss_G = np.mean(losses_G)
            avg_psnr = np.mean(psnr)
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time
            if (i + 1) % 100 == 0:
                print('Iter: (%d/%d) G: %.3f, P: %.3f | Avg: G: %.3f, P: %.3f' %(epoch_iter, len(train_set), current_loss_G, current_psnr, avg_loss_G, avg_psnr), end='\n ')
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch } time: {epoch_time:.2f} seconds")
        model.save_networks('test11')
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")


def test():
    opt.device = 'cuda:0'
    opt.data_root = '/home/li/image_inpaint/data/celeba-hq/test/'  # The location of your testing data
    opt.mask_root = '/home/li/image_inpaint/data/test/cele_test/mask0-0.1/'  # The location of your testing data mask
    testset = MyDataLoader(opt)
    print('Test with %d' % (len(testset)))
    model = MyModel()
    model.initialize(opt)
    model.load_networks('time')  # For irregular mask inpainting
    val_psnr, val_losses_G= [], []
    with torch.no_grad():
        start_time = time.time()
        for i, data in enumerate(testset):
            fname = data['fname'][0]
            iter_start_time = time.time()
            model.set_input(data)
            I_g, I_o, val_loss_G, L_o = model.optimize_parameters(val=True)
            val_p = metrics(I_g, I_o)
            # val_ssim.append(val_s)
            val_psnr.append(val_p)
            # val_mae.append(val_m)
            val_losses_G.append(val_loss_G.detach().item())
            # val_l1_loss.append(val_l1)
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time
            cv2.imwrite('/home/li/TSGDAM/demo/OR/psv/' + fname[:-4] + '.jpg', postprocess(I_o).numpy()[0])
            print('Val (%d/%d) G:%5.4f,  P:%4.2f ,  Time taken: %.6f sec' % (
                i + 1, len(testset), np.mean(val_losses_G), np.mean(val_psnr), iter_time), end='\n')
            # print('Test Time Taken: %.6f sec' % (iter_time), end='\n ')
        end_time = time.time()
        test_time = end_time - start_time
        print(f"test time: {test_time:.2f} seconds")



if __name__ == '__main__':
    if opt.type == 'train':
        train()
    elif opt.type == 'test':
        test()
