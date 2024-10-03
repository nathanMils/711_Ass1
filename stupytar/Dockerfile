FROM jupyter/scipy-notebook:x86_64-ubuntu-22.04

USER root

RUN apt-get update && apt-get install -y nodejs npm && apt-get clean

USER $NB_UID

RUN pip install imblearn jupyterlab 'python-lsp-server[all]' torch torchaudio torchvision \
    && jupyter labextension install @krassowski/jupyterlab-lsp && tabulate

EXPOSE 8888
