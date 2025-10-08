# Dataset

- Description: **datset_name** `nyc` `tky` `ca`

- Please download datasets from this link: https://pan.baidu.com/s/1NEL8o1xuPezxfdU3DCIbsQ?pwd=8mq4.

- Please place them under three folders: **dataset_folder** `Foursquare-NYC` `Foursquare-TKY` `Gowalla-CA` 

# DeepMove, LSTM

- Please enter `DeepMove` folder.

- The data is already prepared in `data/` folder as `nyc.pk` `ca.pk` `tky.pk`.
  
- Train DeepMove:

  ```
   python main.py --data_name {dataset_name}
                  --save_path ../results/{dataset_name}
                  --model_mode attn_local_long
                  --pretrain 0
  ```
- Train LSTM:

  ```
   python main.py --data_name {dataset_name}
                  --save_path ../results/{dataset_name}
                  --model_mode simple
                  --pretrain 0
  ```
  
# MHSA

- Please enter `MHSA` folder.

- Process data
  
  - First, please use `process_{dataset_name}.ipynb` in `dataset/` folder to process each dataset.
  
    The processed dataset will be `train.csv` `valid.csv` `test.csv` in `dataset/{dataset_name}/`.
  
  - Second, for each dataset, please use `preprocessing/foursquare.py` to generate `dataSet_foursquare_{dataset_name}.csv`.
  
    ```
    python preprocessing/foursquare.py --dataset_name {dataset_name}
    ```
    
    The generated data `dataSet_foursquare_{dataset_name}.csv` is in folder `models/MHSA/foursquare/data/`.

- Train MHSA:

  ```
  python models/MHSA/foursquare/main.py config/foursquare/{dataset_name}_transformer.yml
  ```

# LLM-Mob

- This method completely follows https://github.com/Kkhhrr/LLM-Mob-Plus.
- Please enter `MHSA` folder.

- Please unzip data.zip.

- Train LLM-Mob:

  ```
  python llm-mob.py
  ```
  
# LSTPM

- Please enter `LSTPM` folder.

- Please use `nyc_process.ipynb` and `tkyca_process.ipynb` in `dataset/` folder to process each dataset.
    
    The processed results will be `{dataset_name}_cut_one_day.pkl` in `dataset/` folder.

- Train LSTPM:

  ```
  python train.py --dataset_name {dataset_name}
  ```
  
# GETNext

- Please enter `GETNext` folder.

- Please use `build_graph.py` to construct the user-agnostic global trajectory flow map from the training data.

  ```
  python build_graph.py --dataset_name {dataset_name}
  ```

- Train GETNext:

  ```
  python train.py --data-adj-mtx dataset/{dataset_name}/graph_A.csv
                  --data-node-feats dataset/{dataset_name}/graph_X.csv
                  --data-train ../{dataset_folder}/{dataset_name}_train.csv
                  --data-val ../{dataset_folder}/{dataset_name}_val.csv
                  --name {dataset_name}
