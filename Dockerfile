# kaggleのpython環境をベースにする
FROM gcr.io/kaggle-gpu-images/python:v100

# ライブラリの追加インストール
RUN pip install -U pip && \
    pip install efficientnet_pytorch==0.7.1 torchtoolbox==0.1.5 pretrainedmodels==0.7.4 albumentations==0.5.2

RUN conda install -y \
  nodejs

#tqdm
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension \
 && jupyter labextension install @jupyter-widgets/jupyterlab-manager
