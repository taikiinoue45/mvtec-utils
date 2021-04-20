FROM taikiinoue45/mvtec:data AS data
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PATH /opt/conda/bin:/usr/local/cuda-10.2/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
COPY --from=data /data/MVTec/mvtec_anomaly_detection.tar.xz /data/MVTec/mvtec_anomaly_detection.tar.xz
COPY . /root/mvtec-utils
WORKDIR /root/mvtec-utils

RUN set -xe \
        \
        # Install apt requirements
        && apt-get update -y \
        && apt-get install -y --no-install-recommends $(cat docker/requirements_apt.txt) \
        && rm -rf /var/lib/apt/lists/* \
        \
        # Install miniconda and pytorch
        && curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        && chmod +x ~/miniconda.sh \
        && ~/miniconda.sh -b -p /opt/conda \
        && rm ~/miniconda.sh \
        && /opt/conda/bin/conda install -y python=3.6.5 \
        && /opt/conda/bin/conda install pytorch torchvision cudatoolkit=10.2 -c pytorch \
        && /opt/conda/bin/conda clean -ya \
        \
        # Install python requirements and mvtec-utils
        && pip install --no-cache-dir -r docker/requirements_pip.txt \
        && python -m pip install . \
        \
        # Prepare MVTec AD dataset
        && tar -xf /data/MVTec/mvtec_anomaly_detection.tar.xz -C /data/MVTec \
        && python docker/prepare.py \
        && rm -rf /data/MVTec /root/mvtec-utils
