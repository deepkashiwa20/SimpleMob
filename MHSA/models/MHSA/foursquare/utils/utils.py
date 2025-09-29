import yaml
import random, torch, os
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from utils.train import trainNet, test, get_performance_dict
from utils.dataloader import sp_loc_dataset, collate_fn

# 在旧代码中，这里可能导入了 DDP 和 DistributedSampler，在新代码中不再需要
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
from models.MHSA import TransEncoder # 确保这个导入路径是正确的


def load_config(path):
    """
    Loads config file:
    Args:
        path (str): path to the config file
    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for _, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_trainedNets(config, model, train_loader, val_loader, device, log_dir):
    best_model, performance = trainNet(config, model, train_loader, val_loader, device, log_dir=log_dir)
    performance["type"] = "vali"

    return best_model, performance


def get_test_result(config, best_model, test_loader, device):

    return_dict, result_arr_user = test(config, best_model, test_loader, device)

    performance = get_performance_dict(return_dict)
    performance["type"] = "test"

    result_user_df = pd.DataFrame(result_arr_user).T
    result_user_df.columns = [
        "correct@1",
        "correct@3",
        "correct@5",
        "correct@10",
        "rr",
        "ndcg",
        "total",
    ]
    result_user_df.index.name = "user"

    return performance, result_user_df


def get_models(config, device):
    total_params = 0

    if config.networkName == "deepmove":
        # 假设 Deepmove, RNNs, Mobtcast 存在
        # from models.deepmove import Deepmove 
        # model = Deepmove(config=config).to(device)
        raise NotImplementedError("DeepMove model is not fully implemented in the provided files.")
    elif config.networkName == "rnn":
        # from models.rnn import RNNs
        # model = RNNs(config=config, total_loc_num=config.total_loc_num).to(device)
        raise NotImplementedError("RNN model is not fully implemented in the provided files.")
    elif config.networkName == "mobtcast":
        # from models.mobtcast import Mobtcast
        # model = Mobtcast(config=config).to(device)
        raise NotImplementedError("MobTcast model is not fully implemented in the provided files.")
    else:
        # 确保 TransEncoder 已经从 models.MHSA 导入
        model = TransEncoder(config=config, total_loc_num=config.total_loc_num).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 移除 DDP 封装
    # model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False)

    print("Total number of trainable parameters: ", total_params, flush=True)

    return model


def get_dataloaders(config, device): # 移除了 rank 和 world_size 参数
    dataset_train = sp_loc_dataset(
        config.source_root,
        data_type="train",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
    )
    dataset_val = sp_loc_dataset(
        config.source_root,
        data_type="validation",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
    )
    dataset_test = sp_loc_dataset(
        config.source_root,
        data_type="test",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
    )

    # 移除了 DistributedSampler
    # train_sampler = DistributedSampler(...)

    kwds_train = {
        "shuffle": True, # 在非分布式模式下，这里应该设置为 True
        "num_workers": config["num_workers"],
        "drop_last": True,
        "batch_size": config["batch_size"],
        "pin_memory": True if device.type == 'cuda' else False, # 优化GPU数据传输
    }
    kwds_val = {
        "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": True if device.type == 'cuda' else False,
    }
    kwds_test = {
        "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": True if device.type == 'cuda' else False,
    }
    
    # deepmove_collate_fn 在提供的文件中未定义，这里假设使用通用的 collate_fn
    fn = collate_fn

    train_loader = DataLoader(dataset_train, collate_fn=fn, **kwds_train)
    val_loader = DataLoader(dataset_val, collate_fn=fn, **kwds_val)
    test_loader = DataLoader(dataset_test, collate_fn=fn, **kwds_test)

    print(f"Dataloader lengths: Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader