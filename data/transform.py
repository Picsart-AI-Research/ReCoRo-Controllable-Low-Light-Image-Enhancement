from random import uniform
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms import functional as F
from utils.utils import timeit

def get_double_transform(cfg):
    if cfg["name"] == "crop_flip":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*get_norm_mean_std(cfg)),
            transforms.RandomCrop(cfg.crop_size),
            transforms.RandomHorizontalFlip(),
        ]) 
        return transform, transform

    if cfg["name"] == "wo_A_crop":
        transform_A = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*get_norm_mean_std(cfg)),
        ]) 

        transform_B = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*get_norm_mean_std(cfg)),
            transforms.RandomCrop(cfg.crop_size),
        ]) 

        return transform_A, transform_B

    elif cfg["name"] == "crop_w_random_std":
        transform_A = transforms.Compose([
            transforms.ToTensor(),
            RandomSTDNorm(),
            transforms.Normalize(*get_norm_mean_std(cfg)),
            transforms.RandomCrop(cfg.crop_size),
            transforms.RandomHorizontalFlip(),
        ]) 

        transform_B = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*get_norm_mean_std(cfg)),
            transforms.RandomCrop(cfg.crop_size),
            transforms.RandomHorizontalFlip(),
        ]) 
        return transform_A, transform_B

    elif cfg["name"] == "crop_norm":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*get_norm_mean_std(cfg)),
            transforms.RandomCrop(cfg.crop_size),
        ]) 
        return transform, transform

    elif cfg["name"] == "norm":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*get_norm_mean_std(cfg)),
        ]) 
        return transform, transform

    return None, None

def get_double_sync_transform(cfg):
    if cfg["name"] == "sync_crop_flip":
        transform = transforms.Compose([
            SyncRandomCrop(round(cfg.crop_size)),
            SyncToTensor(),
            SyncNormalize(*get_norm_mean_std(cfg)),
            SyncRandomHorizontalFlip(),
            SyncRandomVerticalFlip(),
        ]) 
        
        return transform

    if cfg["name"] == "sync_norm_flip":
        transform = transforms.Compose([
            SyncToTensor(),
            SyncNormalize(*get_norm_mean_std(cfg)),
            SyncRandomHorizontalFlip(),
            SyncRandomVerticalFlip(),
        ]) 
        
        return transform

    if cfg["name"] == "sync_crop":
        transform = transforms.Compose([
            SyncRandomCrop(round(cfg.crop_size)),
        ]) 
        
        return transform

    if cfg["name"] == "random_flips":
        transform = transforms.Compose([
            SyncRandomHorizontalFlip(),
            SyncRandomVerticalFlip(),
        ]) 
        
        return transform

    return None

def get_single_transform(cfg):
    if cfg["name"] == "norm":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*get_norm_mean_std(cfg))
        ]) 
        return transform
    return None

def get_single_sync_transform(cfg):
    if cfg["name"] == "norm_crop":
        transform = transforms.Compose([
            SyncRandomCrop(round(cfg.crop_size)),
            SyncToTensor(),
            SyncNormalize(*get_norm_mean_std(cfg)),
        ]) 
        return transform
    return None

def get_mask_transform(cfg):
    if cfg["name"] == "aug":
        transform = transforms.Compose([
            RandomGaussianBlur(p=0.1),
            RandomGaussianBlur(p=0.1),
            RandomMaxPool(p=0.1),
            RandomMaxPool(p=0.1),
            RandomMinPool(p=0.1),
            RandomMinPool(p=0.1),
        ]) 
        return transform

    if cfg["name"] == "dl_er":
        transform = transforms.Compose([
            MaxPool(),
            MinPool(),
        ]) 
        return transform
    return None



def get_norm_mean_std(cfg):
    mean = [cfg.norm.mean, cfg.norm.mean, cfg.norm.mean]
    std = [cfg.norm.std, cfg.norm.std, cfg.norm.std]
    return mean, std

def get_denorm_mean_std(cfg):
    mean, std = get_norm_mean_std(cfg)
    denorm_mean = [-m/s for m, s in zip(mean, std)]
    denorm_std = [1.0/s for s in std]
    return denorm_mean, denorm_std

def get_denorm_transform(cfg, self_norm=False, min_max_cut=False, to_pil=False):
    transform = []
    if self_norm:
        transform.append(SelfNormalize())
    else:
        transform.append(transforms.Normalize(*get_denorm_mean_std(cfg)))
    
    if min_max_cut:
        transform.append(MinMaxValueCut())

    if to_pil:
        transform.append(transforms.ToPILImage())

    
    return transforms.Compose(transform) 

def get_tensor2pil(cfg):
    return transforms.Compose([
        get_denorm_transform(cfg),
        transforms.ToPILImage()
        ]) 



class SyncResize():
    def __init__(self, scale=0.5):
        self.scale = scale

    def resize(self, imgs):
        for img in imgs:
            width, height = F.get_image_size(img)
            scaled_width = int(width * self.scale)
            scaled_height = int(height * self.scale)
        return [F.resize(img, (scaled_width, scaled_height)) for img in imgs]
    # @timeit("SyncResize")
    def __call__(self, imgs):
        return self.resize(imgs)

    def __repr__(self):
        return self.__class__.__name__ + "()"

class SyncRandomCrop():
    def __init__(self, size=320):
        self.size = size
    # @timeit("SyncRandomCrop")
    def __call__(self, imgs):
        img = imgs[0]
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.size, self.size))
        return [F.crop(img, i, j, h, w) for img in imgs]

    def __repr__(self):
        return self.__class__.__name__ + "()"

class SyncRandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    # @timeit("SyncRandomHorizontalFlip")
    def forward(self, imgs):
        if torch.rand(1) < self.p:
            return [F.hflip(img) for img in imgs]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class SyncRandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    # @timeit("SyncRandomVerticalFlip")
    def forward(self, imgs):
        if torch.rand(1) < self.p:
            return [F.vflip(img) for img in imgs]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class SyncToTensor():
    # @timeit("SyncToTensor")
    def __call__(self, imgs):
        return [F.to_tensor(img) for img in imgs]

    def __repr__(self):
        return self.__class__.__name__ + '()'

class SyncNormalize(nn.Module):
    def __init__(self, mean, std, inplace=False, normalize_masks=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.normalize_masks = normalize_masks
    # @timeit("SyncNormalize")
    def forward(self, imgs):
        norm_imgs = []
        for img_idx, img in enumerate(imgs):
            channels = F.get_image_num_channels(img)
            if channels > 1:
                norm_imgs.append(F.normalize(img, self.mean, self.std, self.inplace))
            else:
                if self.normalize_masks:
                    norm_imgs.append(F.normalize(img, self.mean[0], self.std[0], self.inplace))
                else:
                    norm_imgs.append(img)
        return norm_imgs
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToPILImages(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_pil = transforms.ToPILImage()
    # @timeit("SyncNormalize")
    def forward(self, tensor):
        pil_images = []
        n_images = tensor.shape[0]
        for image_idx in range(n_images):
            pil_images.append(self.to_pil(tensor[image_idx]))
        return pil_images

    def __repr__(self):
        return self.__class__.__name__


class MinMaxValueCut():
    def __init__(self, min=0.0, max=1.0):
        self.min = torch.tensor(min)
        self.max = torch.tensor(max)
    # @timeit("MinMaxValueCut")
    def __call__(self, img):
        img = torch.maximum(img, self.min)
        img = torch.minimum(img, self.max)
        return img

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SelfNormalize():
    def __init__(self, mu=0.5, sigma=0.25):
        self.mu = mu
        self.sigma = sigma
    def __call__(self, img):
        mean = img.mean()
        std = img.std()
        norm_img = (img - mean)/std
        return norm_img*self.sigma + self.mu

    def __repr__(self):
        return self.__class__.__name__ + '(mu={0}, sigma={1})'.format(self.mu, self.sigma)


class RandomSTDNorm(nn.Module):
    def __init__(self, p=0.5, low=2.0, high=4.0):
        super().__init__()
        self.p = p
        self.low = low
        self.high = high
    # @timeit("SyncRandomHorizontalFlip")
    def forward(self, img):
        if torch.rand(1) < self.p:
            std = uniform(self.low, self.high)
            self.std_norm = transforms.Normalize(mean=0.0, std=std)
            return self.std_norm(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomGaussianBlur(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.gb = transforms.GaussianBlur(kernel_size=3)
    # @timeit("SyncRandomHorizontalFlip")
    def forward(self, img):
        if torch.rand(1) < self.p:
            return self.gb(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomMaxPool(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)
    # @timeit("SyncRandomHorizontalFlip")
    def forward(self, img):
        if torch.rand(1) < self.p:
            return self.max_pool(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomMinPool(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)
    def forward(self, img):
        if torch.rand(1) < self.p:
            return -self.max_pool(-img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)
    # @timeit("SyncRandomHorizontalFlip")

    def forward(self, img):
        return self.max_pool(img)

    def __repr__(self):
        return self.__class__.__name__ 

class MinPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, img):
        return -self.max_pool(-img)

    def __repr__(self):
        return self.__class__.__name__
