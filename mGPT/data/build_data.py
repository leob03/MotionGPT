from omegaconf import OmegaConf
from os.path import join as pjoin
from mGPT.config import instantiate_from_config


def build_data(cfg, phase="train"):
    print('leo', phase)
    data_config = OmegaConf.to_container(cfg.DATASET, resolve=True)
    print('leo', data_config)
    data_config['params'] = {'cfg': cfg, 'phase': phase}
    if isinstance(data_config['target'], str):
        print("leo, option 1")
        return instantiate_from_config(data_config)
    elif isinstance(data_config['target'], list):
        print("leo, option 2")
        data_config_tmp = data_config.copy()
        data_config_tmp['params']['dataModules'] = data_config['target']
        data_config_tmp['target'] = 'mGPT.data.Concat.ConcatDataModule'
        return instantiate_from_config(data_config)
