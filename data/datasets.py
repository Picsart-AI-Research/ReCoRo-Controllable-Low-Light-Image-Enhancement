from pathlib import Path
from numpy.random import randint
from hydra.utils import get_original_cwd
from torch.utils import data
import torchvision.transforms as transforms
from .utils import get_image_paths, load_image, get_gray_attention
from .transform import get_double_transform, get_double_sync_transform, get_single_transform

class CombinedDataset(data.Dataset):
    def __init__(self, cfg, dataset_A, dataset_B, a_prefix="ds_A_", b_prefix="ds_B_", alpha_encoder=None, shift_index=False):
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.a_prefix = a_prefix
        self.b_prefix = b_prefix
        self.size_A = len(self.dataset_A)
        self.size_B = len(self.dataset_B)
        if cfg.shift_index:
            self.index_shift = randint(self.size_B)
        else:
            self.index_shift = 0


    def __getitem__(self, index):
        index_A = index % self.size_A
        index_B = (index + self.index_shift) % self.size_B
        item_A = self.dataset_A[index_A]
        item_B = self.dataset_B[index_B]
        item_A = {self.a_prefix + key: val for key, val in item_A.items()}
        item_B = {self.b_prefix + key: val for key, val in item_B.items()}
        item_A.update(item_B)
        return item_A

    def __len__(self):
        return max(self.size_A, self.size_B)

    def name(self):
        return 'CombinedDataset'

class UnpairedMaskDataset(data.Dataset):
    def __init__(self, cfg, alpha_encoder=None):
        self.cfg = cfg
        orig_cwd = get_original_cwd()
        self.images_A_path = Path(f"{orig_cwd}/{self.cfg.images_A}")
        self.gt_masks_path = Path(f"{orig_cwd}/{self.cfg.gt_masks}")
        self.input_masks_path = Path(f"{orig_cwd}/{self.cfg.input_masks}")
        self.images_B_path = Path(f"{orig_cwd}/{self.cfg.images_B}")

        self.images_A_paths = get_image_paths(self.images_A_path)
        self.gt_masks_paths = get_image_paths(self.gt_masks_path)
        self.input_masks_paths = get_image_paths(self.input_masks_path)
        self.images_B_paths = get_image_paths(self.images_B_path)
        
        self.size_A = len(self.images_A_paths)
        self.size_B = len(self.images_B_paths)
        
        self.transform_A, self.transform_B = get_double_transform(cfg.transform)
        self.sync_transform = get_double_sync_transform(cfg.sync_transform)
        self.to_tensor = transforms.ToTensor()
        self.alpha_encoder = alpha_encoder

    def __getitem__(self, index):
        image_A_path = self.images_A_paths[index % self.size_A]
        gt_mask_path = self.gt_masks_paths[index % self.size_A]
        input_mask_path = self.input_masks_paths[index % self.size_A]
        image_B_path = self.images_B_paths[index % self.size_B]

        image_A = load_image(image_A_path)
        gt_mask = load_image(gt_mask_path, mode="L")
        input_mask = load_image(input_mask_path, mode="L")
        image_B = load_image(image_B_path)

        if self.sync_transform is not None:
            image_A, gt_mask, input_mask = self.sync_transform([image_A, gt_mask, input_mask])

        if self.transform_B is not None:
            image_B = self.transform_B(image_B)

    
        item = {
            "image_A": image_A, 
            "gt_mask": gt_mask, 
            "input_mask": input_mask, 
            "image_A_path": str(image_A_path), 
            "image_B": image_B, 
            "image_B_path": str(image_B_path)
        }

        if self.alpha_encoder is not None:
            alpha, alpha_vec = self.alpha_encoder.sample(self.cfg.alpha)
            item.update({
                "alpha": alpha, 
                "alpha_vec": alpha_vec
            })

        return item

    def __len__(self):
        return max(self.size_A, self.size_B)

    def name(self):
        return 'UnpairedMaskDataset'

class PairedMaskDataset(data.Dataset):
    def __init__(self, cfg, alpha_encoder=None):
        self.cfg = cfg
        orig_cwd = get_original_cwd()
        self.images_A_path = Path(f"{orig_cwd}/{self.cfg.images_A}")
        self.gt_masks_path = Path(f"{orig_cwd}/{self.cfg.gt_masks}")
        self.input_masks_path = Path(f"{orig_cwd}/{self.cfg.input_masks}")
        self.images_B_path = Path(f"{orig_cwd}/{self.cfg.images_B}")

        self.images_A_paths = get_image_paths(self.images_A_path)
        self.gt_masks_paths = get_image_paths(self.gt_masks_path)
        self.input_masks_paths = get_image_paths(self.input_masks_path)
        self.images_B_paths = get_image_paths(self.images_B_path)
        
        self.size_A = len(self.images_A_paths)
        self.size_B = len(self.images_B_paths)

        self.transform_A, self.transform_B = get_double_transform(cfg.transform)
        self.sync_transform = get_double_sync_transform(cfg.sync_transform)
        self.to_tensor = transforms.ToTensor()
        self.alpha_encoder = alpha_encoder

    def __getitem__(self, index):
        image_A_path = self.images_A_paths[index % self.size_A]
        gt_mask_path = self.gt_masks_paths[index % self.size_A]
        input_mask_path = self.input_masks_paths[index % self.size_A]
        image_B_path = self.images_B_paths[index % self.size_B]

        image_A = load_image(image_A_path)
        gt_mask = load_image(gt_mask_path, mode="L")
        input_mask = load_image(input_mask_path, mode="L")
        image_B = load_image(image_B_path)

        if self.sync_transform is not None:
            image_A, gt_mask, input_mask, image_B = self.sync_transform([image_A, gt_mask, input_mask, image_B])
        
        mask_A = self.to_tensor(mask_A)
        mask_B = self.to_tensor(mask_B)
        if self.transform_A is not None:
            image_A = self.transform_A(image_A)
        if self.transform_B is not None:
            image_B = self.transform_B(image_B)

    
        item = {
            "image_A": image_A, 
            "gt_mask": gt_mask, 
            "input_mask": input_mask, 
            "image_A_path": str(image_A_path), 
            "image_B": image_B, 
            "image_B_path": str(image_B_path)
        }

        if self.alpha_encoder is not None:
            alpha, alpha_vec = self.alpha_encoder.sample(self.cfg.alpha)
            item.update({
                "alpha": alpha, 
                "alpha_vec": alpha_vec
            })

        return item

    def __len__(self):
        return max(self.size_A, self.size_B)

    def name(self):
        return 'PairedMaskDataset'

class PairedAugMaskDataset(data.Dataset):
    def __init__(self, cfg, alpha_encoder=None):
        self.cfg = cfg
        orig_cwd = get_original_cwd()
        self.images_A_path = Path(f"{orig_cwd}/{self.cfg.images_A}")
        self.gt_masks_path = Path(f"{orig_cwd}/{self.cfg.gt_masks}")
        self.input_aug_masks_paths = [Path(f"{orig_cwd}/{dir_name}") for dir_name in cfg.aug_masks]
        self.images_B_path = Path(f"{orig_cwd}/{self.cfg.images_B}")

        self.images_A_paths = get_image_paths(self.images_A_path)
        self.gt_masks_paths = get_image_paths(self.gt_masks_path)
        self.input_aug_masks_pathss = [get_image_paths(path) for path in self.input_aug_masks_paths]
        self.images_B_paths = get_image_paths(self.images_B_path)
        
        self.size_A = len(self.images_A_paths)
        self.size_B = len(self.images_B_paths)
        self.n_gt_maskugs = len(self.input_aug_masks_paths)
        
        self.transform_A, self.transform_B = get_double_transform(cfg.transform)
        self.sync_transform = get_double_sync_transform(cfg.sync_transform)
        self.to_tensor = transforms.ToTensor()
        self.alpha_encoder = alpha_encoder

    # @timeit("DoubleUnpairedDataset")
    def __getitem__(self, index):
        image_A_path = self.images_A_paths[index % self.size_A]
        gt_mask_path = self.gt_masks_paths[index % self.size_A]
        aug_mask_idx = randint(self.n_gt_maskugs )
        input_mask_path = self.input_aug_masks_pathss[aug_mask_idx][index % self.size_A]
        image_B_path = self.images_B_paths[index % self.size_B]

        image_A = load_image(image_A_path)
        gt_mask = load_image(gt_mask_path, mode="L")
        input_mask = load_image(input_mask_path, mode="L")
        image_B = load_image(image_B_path)

        if self.sync_transform is not None:
            image_A, gt_mask, input_mask, image_B = self.sync_transform([image_A, gt_mask, input_mask, image_B])

        if self.transform_A is not None:
            image_A = self.transform_A(image_A)
        if self.transform_B is not None:
            image_B = self.transform_B(image_B)

    
        item = {
            "image_A": image_A, 
            "gt_mask": gt_mask, 
            "input_mask": input_mask, 
            "image_A_path": str(image_A_path), 
            "image_B": image_B, 
            "image_B_path": str(image_B_path)
        }

        if self.alpha_encoder is not None:
            alpha, alpha_vec = self.alpha_encoder.sample(self.cfg.alpha)
            item.update({
                "alpha": alpha, 
                "alpha_vec": alpha_vec
            })

        return item

    def __len__(self):
        return max(self.size_A, self.size_B)

    def name(self):
        return 'PairedAugMaskDataset'

class MaskDataset(data.Dataset):
    def __init__(self, cfg, alpha_encoder=None):
        self.cfg = cfg
        orig_cwd = get_original_cwd()
        self.images_path = Path(f"{orig_cwd}/{self.cfg.images}")
        self.input_masks_path = Path(f"{orig_cwd}/{self.cfg.input_masks}")

        self.images_paths = get_image_paths(self.images_path)
        self.input_masks_paths = get_image_paths(self.input_masks_path)
        
        self.size = len(self.images_paths)

        self.transform = get_single_transform(cfg.transform)
        self.to_tensor = transforms.ToTensor()
        self.alpha_encoder = alpha_encoder

    def __getitem__(self, index):
        image_path = self.images_paths[index % self.size]
        input_mask_path = self.input_masks_paths[index % self.size]

        image = load_image(image_path)
        input_mask = load_image(input_mask_path, mode="L")

        if self.transform is not None:
            image = self.transform(image)
        input_mask = self.to_tensor(input_mask)

        item = {
            "image_A": image, 
            "gt_mask": input_mask, 
            "input_mask": input_mask, 
            "image_A_path": str(image_path), 
            "image_B": image, 
            "image_B_path": str(image_path)
        }

        if self.alpha_encoder is not None:
            alpha, alpha_vec = self.alpha_encoder.sample(self.cfg.alpha)
            item.update({
                "alpha": alpha, 
                "alpha_vec": alpha_vec
            })

        return item
        
    def __len__(self):
        return self.size

    def name(self):
        return 'MaskDataset'