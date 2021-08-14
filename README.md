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
$ docker-compose up --build -d
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
Plese download data to `./data` from https://www.kaggle.com/takoihiraokazu/commonlit-fold and unzip it. This is the data for cross validation, and is the result of re-running [here](https://www.kaggle.com/abhishek/step-1-create-folds).


 # Train
 `$ sh bin/train.sh` 
 
 Please note that the results of ex131.py(deberta-v2-xlarge), ex194.py(deberta-v2-xlarge), ex216.py(deberta-v2-xxlarge) and ex407.py(funnel-transformer-large) will change with each train. Also, I did not fix the seed in mlm_roberta_base.py during the competition period, so if you want to reproduce ex237.py, please use  [this pretrained model](https://www.kaggle.com/takoihiraokazu/clrp-roberta-base-mlm) instead of output of mlm_roberta_base.py. I have also listed below the hardware numbers where I performed each training during the competition.
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

# Ensemble & Post process
`$ sh bin/ensemble_postprocess.sh` 

The ensemble weights were adjusted based on the output of ensemble.py and by looking at the Public Score.
The PostProcess coefficients and thresholds were also adjusted based on the output of postprocess.py and by looking at the Public Score.
The final weights, coefficientsare and thresholds as follows. The public and private scores for each model are also listed.
### ensebmle
| exp | model| Public| Private| weight |
| ---- | ---- | ---- | ---- |---- |
| ex015 | roberta base -> svr |0.476 |0.478 | 0.020 |
| ex015 | roberta base -> ridge |0.475 |0.476 | 0.020 |
| ex072 | roberta large| 0.463| 0.466| 0.088|
| ex107 | bart large | 0.469| 0.471| 0.090|
| ex182 | deberta large | 0.460| 0.463| 0.230|
| ex190 | electra large | 0.470| 0.471| 0.050|
| ex194 | deberta v2 xlarge | 0.466 | 0.467| 0.050|
| ex216 | deberta v2 xxlarge | 0.465| 0.466| 0.140|
| ex237 | roberta base(with mlm) | 0.476| 0.473| 0.040|
| ex272 | funnel large base | 0.471| 0.473| 0.050|
| ex292 | mpnet base | 0.470| 0.473| 0.130|
| ex384 | muppet roberta large | 0.466| 0.468| 0.022|
| ex407 | funnel large | 0.464| 0.471| 0.110|
| ex429 | gpt2 medium | 0.478| 0.482| 0.170|
| ex434 | t5 large | 0.498| 0.494| -0.110|
| ex448 + ex450 | albert xxlarge v2 | 0.467| 0.471| 0.120|
| ex465 | electra base | 0.482| 0.483| -0.170|
| ex497 | bert base uncased | 0.506| 0.497| -0.140|
| ex507 | distilbart cnn 12 6 | 0.479| 0.477| 0.090|
### post process
-	Predictions >= 0.3 -> Predictions * 1.07
-	0.3 > Predictions >= 0 -> Predictions * 1.2
-	0 > Predictions >= -0.7 -> Predictions * 0.97485814
-	-0.7 > Predictions >= -0.9 -> Predictions * 1.01
-	-0.9 > Predictions >= -2.0 -> Predictions * 1.02150304
-	-2.0 > Predictions -> Predictions * 1.02764047

# Predict
`$ sh bin/predict.sh`
