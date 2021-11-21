# SpeedyRec on MIND dataset
This example walks through the training and prediction of SpeedyRec on MIND dataset.  
After cloning this repo, you need go to this dictionary `cd ./speedy_mind` and conduct experiments with following commands:

## Requirements
```bash
Python==3.6
transformers==4.6.0
tensforlow==1.15
scikit-learn==0.23
```

## Preparing Data
Download data from MIND [link](https://msnews.github.io/) abd decompress these files.

Script `data_generation.py` can help you to generate the data files which meet the need of SpeedyRec:
```
python data_generation.py --raw_data_path {path to your decompressed data}
```

## Training 
```
python train.py \
--pretrained_model_path ./unilm_ckpt/ \
--pretreained_model unilm \
--root_data_dir ./data/speedy_data/ \
--num_hidden_layers 8 \
--world_size 4 \
--lr 1e-4 \
--pretrain_lr 8e-6 \
--warmup True \
--schedule_step 120000 \
--warmup_step 1000 \
--batch_size 42 \
--npratio 4 \
--epochs 3 \
--beta_for_cache 0.002 \
--max_step_in_cache 2 \
--news_attributes title \
--savename speedyrec_mind 
```
The model will be saved to `./save_ckpts/`, and validation will be conducted after each epoch.   
The default pretrained model is unilm, you can download it from this link. For other pretrained model, you need set `--pretreained_model==others` and give a new path for `--pretreained_model_path`
(like `roberta-base` and `microsoft/deberta-base`, which needs to be supported by transformers).



## Prediction
Run prediction using saved checkpoint:
```
python submission.py \
--pretrained_model_path ./unilm_ckpt/ \
--pretreained_model unilm \
--root_data_dir ./data/speedy_data/ \
--num_hidden_layers 8 \
--load_ckpt_name {path to your saved model} \
--news_attributes title \
--batch_size 256 
```
It will creates a zip file:`predciton.zip`, which can be submitted to the leaderboard of MIND directly.

