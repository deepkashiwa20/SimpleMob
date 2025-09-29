import torch, os
import numpy as np
import argparse
import pandas as pd
from datetime import datetime
import json

from easydict import EasyDict as edict

from utils.utils import load_config, setup_seed, get_trainedNets, get_test_result, get_dataloaders, get_models

def main():
    setup_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, nargs="?", help=" Config file path.", default="config/foursquare/transformer.yml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    time_now = int(datetime.now().timestamp())
    networkName = f"{config.dataset}_{config.networkName}"
    log_dir = os.path.join(config.save_root, f"{networkName}_{config.previous_day}_{str(time_now)}_single")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "conf.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    train_loader, val_loader, test_loader = get_dataloaders(config, device)

    model = get_models(config, device)

    model, perf_val = get_trainedNets(config, model, train_loader, val_loader, device, log_dir)

    perf_test, test_df = get_test_result(config, model, test_loader, device)
    test_df.to_csv(os.path.join(log_dir, "user_detail.csv"))

    result_df = pd.DataFrame([perf_val, perf_test])
    print("\nValidation and Test Results:")
    print(result_df)
    
    filename = os.path.join(
        config.save_root,
        f"{config.dataset}_{config.networkName}_results_{str(time_now)}.csv",
    )
    result_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main()