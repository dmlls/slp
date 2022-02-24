from omegaconf import OmegaConf
from omegaconf. dictconfig import DictConfig

config: DictConfig = OmegaConf.load('../config/config.yaml')
