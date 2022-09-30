from time import time

from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import numpy as np
from tqdm import tqdm 

from data import get_data
from models.engan import get_model
from utils.visualize import Visualizer
from models.losses import SSIMLoss
from data.transform import get_denorm_transform, ToPILImages

@hydra.main(config_path="./conf", config_name="config_base_train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device)
    dataset, dataloader, eval_dataset, eval_dataloader = get_data(cfg.data)
    model = get_model(cfg, device)
    visualizer = Visualizer(cfg)
    ssim_metric = SSIMLoss(data_range=1.0)
    denorm = get_denorm_transform(cfg.data.transform, min_max_cut=True)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm(range(cfg.epochs)):
        update_discriminators = (epoch // 5) % 2 == 0
        tqdm.write(f"update_discriminators: {update_discriminators}")
        bt = []
        for batch_idx, data in enumerate(dataloader):
            iteration = batch_idx + epoch * len(dataloader)
            return_images = batch_idx == 0
            bt_start = time()
            losses, images = model.step(data, return_images=return_images, epoch=epoch, batch_idx=batch_idx)
            bt.append(time() - bt_start)
            if return_images:
                visualizer.add_images(data, images, epoch)
                # visualizer.add_weights_histogram(epoch, model)
            visualizer.add_scalars(losses, iteration)
        tqdm.write(f"losses {epoch}: {losses}")
        if epoch % cfg.save_freq == 0:
            model.save(epoch=epoch)
        tqdm.write(f"epoch time: {sum(bt):0.3f}")
        model.update_learning_rate(epoch)
   
    model.save(epoch=epoch)
    for _, net in model.networks.items():
        net.eval()
    ssims = []
    for eval_data in eval_dataloader:
        results = model.predict(eval_data)
        fake_B = denorm(results["fake_B"])
        real_B = denorm(eval_data["image_B"])
        ssims.append((1.0 - ssim_metric(fake_B, real_B)).item())
    tqdm.write(f"ssim: {np.mean(ssims):0.3f}")

if __name__ == "__main__":
    main()