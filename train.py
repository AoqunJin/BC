import warnings
warnings.filterwarnings("ignore")

from config import get_LCBCConfig, get_RT1Config, get_DiffusionConfig, get_InverseConfig
from data import get_dataloader_split
from ray_trainer import train_ray, tune_dist_ray


def main(train_model = 'lcbc'):
    if train_model == 'lcbc':
        config = get_LCBCConfig()
        from models.lcbc import (
            get_model, forward_fn
        )
    elif train_model == 'rt1':
        config = get_RT1Config()
        from models.rt1 import (
            get_model, forward_fn
        )    
    elif train_model == 'diffusion':
        config = get_DiffusionConfig()
        from models.diffusion import (
            get_model, forward_fn
        )
    elif train_model == 'inverse':
        config = get_InverseConfig()
        from models.inverse import (
            get_model, forward_fn
        )
    
    config["get_dataloader"] = get_dataloader_split
    config["get_model"] = get_model
    config["forward_fn"] = forward_fn
    
    if config["tune"]:
        tune_dist_ray(config=config)
    else:
        train_ray(config=config)
    
if __name__ == '__main__':
    main('inverse')
    