import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="train")
def train(config: DictConfig):
    print(config)
