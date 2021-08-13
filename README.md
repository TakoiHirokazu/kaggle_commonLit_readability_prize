# Kaggle CommonLit Readability Prize 2nd place solution

This is repository of the 2nd place solution of [Kaggle CommonLit Readability Prize](https://www.kaggle.com/c/commonlitreadabilityprize).  
The discription of this solution is available [here](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258328).  
The prediction notebook in the competition is available [here](https://www.kaggle.com/takoihiraokazu/lb-ensemble-add-electra-base-bert-base2-t5-diba2).  
A partially revised prediction notebook is [here](https://www.kaggle.com/takoihiraokazu/final-sub1).

# Hardware
I used three different machines.

1. Google Cloud Platform
- Debian 10.10
- n1-highmem-8 (vCPU x 8, memory 52 GB)
- 1 x NVIDIA Tesla V100

2.  Google Cloud Platform
- Ubuntu 18.04
- a2-highgpu-1g (vCPU x 12, memory 85 GB)
- 1 x NVIDIA Tesla A100

3. kaggle notebook

# Environment(except for kaggle notebook)
```
$ docker-compose up --build
$ docker exec -it commonlit bash
```

# Data download
Plese download data to `./data` from https://www.kaggle.com/c/commonlitreadabilityprize/data and unzip it.

# Model download
I used several pretrained models that are publicly available in the kaggle dataset. Please download it to the following directories and unzip it.
 - `./models/roberta` : https://www.kaggle.com/maroberti/roberta-transformers-pytorch
 - `./models/bart` : https://www.kaggle.com/xhlulu/bart-models-hugging-face-model-repository
 - `./models/electra` : https://www.kaggle.com/xhlulu/electra 
 - `./models/deberta` : https://www.kaggle.com/xhlulu/deberta
 - `./models/xlnet` : https://www.kaggle.com/irustandi/xlnet-pretrained-models-pytorch

 # Prerocess
 ここにコマンド

 # Train
 `$ sh bin/train.sh` 

 Training results vary depending on the hardware, so if you want to reproduce the results, you need to change the hardware according to the model.
 However, please note that the results for ex131.py(deberta-v2-xlarge), ex194.py(deberta-v2-xlarge), ex216.py(deberta-v2-xxlarge) and ex407.py(funnel-transformer-large) could not be reproduced even if the hardware is the same and the seed is fixed. Also, I hadn't fixed the seed in the pretrain of roberta-base, so if you want to reproduce ex237.py, please use  [here](https://www.kaggle.com/takoihiraokazu/clrp-roberta-base-mlm). 
 
| exp | Hardware|
| ---- | ---- | 
| ex014.py | 3 | 
| ex015.py | 3 | 
| ex064.py | 1 |
| ex072.py | 1 |
| ex084.py | 1 |
| ex094.py | 1 |
| ex107.py | 1 |
| ex131.py | 2 |
| ex182.py | 2 |
| ex190.py | 2 |
| ex194.py | 2 |
| ex216.py | 2 |
| mlm_roberta_base.py | 3 | 
| ex237.py | 3 |
| ex272.py | 2 |
| ex292.py | 2 |
| ex384.py | 2 |
| ex407.py | 2 |
| ex429.py | 2 |
| ex434.py | 2 |
| ex448.py | 2 |
| ex450.py | 2 |
| ex465.py | 2 |
| ex497.py | 2 |
| ex507.py | 2 |

# Ensemble & Postprocess
ここにコマンド

The ensemble weights were adjusted based on the output of the optimization and by looking at the Public Score.
The PostProcess coefficients were also adjusted based on the optimization output and by looking at the Public Score.
 # Predict

