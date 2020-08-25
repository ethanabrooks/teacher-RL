FROM nvcr.io/nvidia/pytorch:19.10-py3

RUN apt-get update && apt-get install -y rsync  && rm -rf /var/lib/apt/lists/*

COPY ./docker-environment.yml /tmp/environment.yml
RUN conda env update -f /tmp/environment.yml \
    && conda clean --all -y

RUN echo "source activate base" >> /root/.bashrc
ENV PATH /opt/conda/envs/jax/bin:$PATH
RUN pip install -U ipdb

WORKDIR "/root"
COPY . .
