# Save as Dockerfile.jlab
FROM docker.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
# Or another 2.x CUDA12.x runtime tag you prefer

ARG NB_USER=app
ARG NB_UID=1000
ARG NB_GID=1000

# basics for JupyterLab Git extension + proper signal handling
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git tini \
    && rm -rf /var/lib/apt/lists/*

# JupyterLab + extras
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir \
      jupyterlab==4.* \
      jupyterlab-git==0.51.* \
      ipykernel ipywidgets jupyterlab_widgets jupyter-resource-usage \
 && jupyter server extension enable --py jupyterlab_git

# non-root user
RUN groupadd -g ${NB_GID} ${NB_USER} \
 && useradd -m -s /bin/bash -u ${NB_UID} -g ${NB_GID} ${NB_USER}

USER ${NB_USER}
WORKDIR /home/${NB_USER}/work

EXPOSE 8888
ENV JUPYTER_TOKEN=""

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token=${JUPYTER_TOKEN}"]
