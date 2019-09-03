FROM tensorflow/tensorflow:1.13.2-gpu-py3-jupyter

# Install JupyterLab
RUN pip install jupyterlab && jupyter serverextension enable --py jupyterlab

RUN pip install pandas
RUN pip install scipy

# Dark Theme
COPY conf/jupyter_cfg/themes.jupyterlab-settings /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

WORKDIR /workdir
EXPOSE 80