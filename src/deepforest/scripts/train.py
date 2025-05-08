import hydra
from omegaconf import DictConfig, OmegaConf

from deepforest.main import deepforest


@hydra.main(version_base=None, config_path="pkg://deepforest.conf", config_name="config")
def main(cfg: DictConfig) -> None:

    m = deepforest(config=cfg)
    m.trainer.fit(m)


if __name__ == "__main__":
    main()
