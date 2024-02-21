import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import optuna
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate, call
from src.util.eval import linear_probe
from src import train

def optuna_override(cfg, trial):
    dotlist = []
    for param_name in cfg.paramspace.keys():
        val = call(cfg.paramspace[param_name], self=trial, name=param_name)
        dotlist.append(f'{param_name}={val}')
    
    suggested_cfg = OmegaConf.from_dotlist(dotlist)
    newcfg = OmegaConf.merge(cfg, suggested_cfg)
    return newcfg

@hydra.main(version_base=None, config_path='../config', config_name='paramsearch')
def main(config: DictConfig):
    def objective(trial):
        newconfig = optuna_override(config, trial)
        model = train.main(newconfig)
        train_loader, test_loader = train.get_loaders(newconfig)
        invariance = model.__class__.__name__ == ("LitMSPAE" or "LitEquivariantAE")
        n_samples = 40 if config.test else None
        return linear_probe(train_loader.dataset, test_loader.dataset, model, model.device, n_samples, invariance, C=0.1)
    
    study = instantiate(config.optuna.create_study)
    #study.optimize(objective, config.optuna.optimize.n_trials)
    call(config.optuna.optimize, self=study, func=objective)

if __name__ == '__main__':
    main()