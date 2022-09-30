from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from data.transform import get_denorm_transform, ToPILImages
import torchvision.transforms as transforms
from torchvision.utils import save_image
from hydra.utils import get_original_cwd

def torch_imshow(batch, batchsize=64, width=8, height=8, title=None):
    batch = batch[:batchsize].cpu()
    plt.figure(figsize=(width, height))
    plt.axis("off")
    if title is not None:
        plt.title(title, fontsize=12)
    grid = make_grid(batch, padding=2, normalize=True)
    grid = np.transpose(grid, (1,2,0))
    plt.imshow(grid)
    plt.savefig("torch_imshow.png")    
    
def save_batch(batch, filename):
    save_image(make_grid(batch), filename)

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

class Visualizer():
    def __init__(self, cfg):
        self.writer = SummaryWriter(".")
        self.cfg = cfg
        self.denorm = get_denorm_transform(cfg.data.transform, min_max_cut=True)
        self.attention_denorm = get_denorm_transform(cfg.data.transform, self_norm=True, min_max_cut=True)
        self.to_pil = transforms.ToPILImage()
        self.to_pil_images = ToPILImages()
        self.orig_cwd = get_original_cwd()
        self.results_path = Path("./results")
        self.results_path.mkdir(exist_ok=True)


    def add_scalars(self, losses, iteration):
        if self.cfg.model.type in ["EnlightenGAN", "RealEnlightenGAN", "BranchesGAN", "SimpleGAN", "ZeroGAN", "MaskGAN", "LiifHintMaskGAN", "ZeroBaseGAN"]:
            self.writer.add_scalars('Losses/train', losses, iteration)
        elif self.cfg.model.type in ["LIIFEnlightenGAN", "FiLMEnlightenGAN", "LIIFMaskEnlightenGAN", "LIIFHingeMaskEnlightenGAN", "SPADEEnlightenGAN", "LIIFSPADEEnlightenGAN", "CoarseMaskEnlightenGAN", "HardMaskEnlightenGAN", "CoarseSpadeMaskEnlightenGAN", "AuxSpadeCoarseMaskEnlightenGAN", "ReCoRoGAN"]:
            self.writer.add_scalars('Losses/train', losses[0], iteration)
            self.writer.add_scalars('Losses/train', losses[1], iteration)
        elif self.cfg.model.type in ["LIIFPercEnlightenGAN", "FiLMPercEnlightenGAN"]:
            self.writer.add_scalars('Losses/train/base', losses[0], iteration)
            self.writer.add_scalars('Losses/train/alpha', losses[1], iteration)
        else:
            raise "Unknown visualisation config"

    def add_images(self, data, images, iteration):
        if self.cfg.model.type in ["EnlightenGAN"]:
            self.writer.add_images('0 input', self.denorm(data["image_A"][:4]), iteration)
            self.writer.add_images('1 enhanced', self.denorm(images["fake_B"][:4]), iteration)  
            self.writer.add_images('2 normal light', self.denorm(data["image_B"][:4]), iteration)
            self.writer.add_images('3 attention', self.denorm(data["A_gray"].expand(-1,3,-1,-1)[:4]), iteration)
            self.writer.add_images('4 latent', self.denorm(images["latent"][:4]), iteration)            
            self.writer.add_images('5 fake_B_patches', self.denorm(images["fake_B_patches"][:4][0]), iteration)            
            self.writer.add_images('6 real_B_patches', self.denorm(images["real_B_patches"][:4][0]), iteration)           
        elif self.cfg.model.type == "ReCoRoGAN":
            self.writer.add_images('0 input', self.denorm(data["ds_A_image_A"][:4]), iteration)
            self.writer.add_images('1 enhanced', self.denorm(images[0]["fake_B"][:4]), iteration)  
            self.writer.add_images('1 latent', self.denorm(images[0]["latent"][:4]), iteration)            
            self.writer.add_images('2 normal light', self.denorm(data["ds_A_image_B"][:4]), iteration)
            self.writer.add_images('3 pred_fake', images[0]["pred_fake"].expand(-1,3,-1,-1)[:4], iteration)
            self.writer.add_images('4 pred_real', images[0]["pred_real"].expand(-1,3,-1,-1)[:4], iteration)
            self.writer.add_images('5 fake_B_patches', self.denorm(images[0]["fake_B_patches"][:4][0]), iteration)            
            self.writer.add_images('6 real_B_patches', self.denorm(images[0]["real_B_patches"][:4][0]), iteration)     

            low_lights = self.denorm(data["ds_B_image_A"][:4])
            self.writer.add_images('7 input', low_lights, iteration)
            self.writer.add_images('7 fine mask', data["ds_B_gt_mask"][:4], iteration)
            self.writer.add_images('7 coarse mask', data["ds_B_input_mask"][:4], iteration)
            enhanced = self.denorm(images[1]["fake_B"][:4])
            self.writer.add_images(f"8 enhanced", enhanced, iteration) 
            self.writer.add_images('8 latent', self.denorm(images[1]["latent"][:4]), iteration)             
            self.writer.add_images('9 normal light', self.denorm(data["ds_B_image_B"][:4]), iteration)
            self.writer.add_images('10 pred_fake', images[1]["pred_fake"].expand(-1,3,-1,-1)[:4], iteration)
            self.writer.add_images('12 latent', self.denorm(images[1]["latent"][:4]), iteration)                
        else:
            raise "Unknown visualisation config"

    def save_images(self, data, images):
        if self.cfg.model.type == "EnlightenGAN":
            img_path_relative = Path(data["A_path"][0]).relative_to(Path(f"{self.orig_cwd}/{self.cfg.data.dir}"))
            save_path = self.results_path / img_path_relative
            save_path.parent.mkdir(exist_ok=True, parents=True)
            self.to_pil(self.denorm(images["fake_B"][0])).save(save_path.parent / f"{save_path.stem}_enhanced.png")
            self.to_pil(self.denorm(images["latent"][0])).save(save_path.parent / f"{save_path.stem}_latent.png")
        elif self.cfg.model.type == "ReCoRoGAN":
            for idx in range(len(data["A_path"])):
                img_path_relative = Path(data["A_path"][idx]).relative_to(Path(f"{self.orig_cwd}/{self.cfg.data.dir}"))
                save_path = self.results_path / img_path_relative
                save_path.parent.mkdir(exist_ok=True, parents=True)
                self.to_pil(self.denorm(data["image_A"][idx])).save(save_path.parent / f"{save_path.stem}.png")
                self.to_pil(self.denorm(images["fake_B"][idx])).save(save_path.parent / f"{save_path.stem}_enhanced.png")
                self.to_pil(self.denorm(images["latent"][idx])).save(save_path.parent / f"{save_path.stem}_latent.png")
                self.to_pil((data["gt_mask"])[idx]).save(save_path.parent / f"{save_path.stem}_mask.png")
        else:
            raise "Unknown visualisation config"
