from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
import torch
from data import get_data
from models.engan import get_model
from pathlib import Path
from utils.visualize import Visualizer
from tqdm import tqdm

@hydra.main(config_path="./conf", config_name="config__base_test")
def main(cfg: DictConfig):
    print(cfg)
    # default `log_dir` is "runs" - we'll be more specific here

    device = torch.device(cfg.device)
    dataset, dataloader = get_data(cfg.data)
    model = get_model(cfg, device, is_train=False)

    orig_cwd = get_original_cwd()
    checkpoints = Path(f"{orig_cwd}/{cfg.model.checkpoints}")
    model.load(checkpoints=checkpoints, epoch=cfg.model.load_epoch, label=cfg.model.label)
    visualizer = Visualizer(cfg)
    
    for data in tqdm(dataloader):
        results = model.predict(data)
        visualizer.save_images(data, results)
            # img_path_relative = Path(img_path).relative_to(Path(f"{orig_cwd}/{cfg.data.dir}"))
            # save_path = results_path / img_path_relative
            # save_path.parent.mkdir(exist_ok=True, parents=True)
            # img.save(save_path)

if __name__ == "__main__":
    main()