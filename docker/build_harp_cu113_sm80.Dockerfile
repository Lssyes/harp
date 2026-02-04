FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ARG SM=80

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace


RUN cp /etc/apt/sources.list /etc/apt/sources.list.backup && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list  && \
    apt-get update  && \
    apt-get install -y openssh-server wget git vim net-tools iftop coinor-cbc build-essential libibverbs-dev devscripts debhelper fakeroot


## setup ssh
## Make sure that you have prepared your ssh keys in docker/ssh_key/id_rsa and docker/ssh_key/authorized_keys,
## because we use script built on ssh to control ray cluster
COPY docker/ssh_key /root/.ssh/
RUN chmod 700 ~/.ssh && \
    chmod 600 ~/.ssh/id_rsa && \
    service ssh restart


## build nccl 2.19.3 from source
RUN cd /workspace && \
    git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    git checkout v2.19.3-1 && \
    make -j src.build NVCC_GENCODE="-gencode=arch=compute_${SM},code=sm_${SM}" && \
    cp /workspace/nccl/build/lib/libnccl.so* /usr/lib/x86_64-linux-gnu/ && \
    cp /workspace/nccl/build/lib/libnccl_static.a  /usr/lib/x86_64-linux-gnu/ && \
    cd /usr/lib/x86_64-linux-gnu/ && \
    rm libnccl.so.2 && \
    ln -s libnccl.so.2.19.3 libnccl.so.2 && \
    rm /usr/include/nccl*.h && \
    cp /workspace/nccl/build/include/nccl*.h /usr/include/


## install python3.8 and pip
RUN apt-get install -y python3.8 python3.8-dev python3.8-venv python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    ln -s /usr/bin/python3.8 /usr/bin/python  && \
    python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --upgrade pip


## build cupy-cuda11x from source
RUN cd /workspace && \
    git clone https://github.com/cupy/cupy.git && \
    cd cupy && \
    git checkout v12 && \
    git submodule update --init --recursive && \
    pip install .


## build jaxlib
COPY docker/build_jaxlib /workspace/build_jaxlib
COPY third_party/alpa/third_party/jax /workspace/jax
COPY third_party/alpa/third_party/tensorflow-alpa /workspace/tensorflow-alpa
COPY harp_patches/tensorflow-alpa/* /workspace/tensorflow-alpa/tensorflow/compiler/xla/service/gpu/


RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    ln -sf /usr/local/cuda-11.3/lib64/stubs/libcuda.so /usr/local/cuda-11.3/lib64/stubs/libcuda.so.1 && \
    pip install numpy==1.20.0 setuptools wheel six auditwheel && \
    apt-get -y install gcc-7 g++-7 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100

RUN cd /workspace/build_jaxlib && \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH && \
    python build/build.py \
        --python_bin_path=/usr/bin/python3.8 \
        --enable_cuda \
        --cuda_path=/usr/local/cuda-11.3 \
        --cudnn_path=/usr/lib/x86_64-linux-gnu \
        --cuda_version=11.3 \
        --cudnn_version=8 \
        --cuda_compute_capabilities="sm_${SM},compute_${SM}" \
        --dev_install \
        --bazel_options=--config=linux \
        --bazel_options=--override_repository=org_tensorflow=/workspace/tensorflow-alpa \
        --bazel_options=--action_env=TF_CUDA_PATHS=/usr/local/cuda-11.3,/usr/lib/x86_64-linux-gnu,/usr && \
    cd dist && \
    pip install -e .


## install alpa env
RUN mkdir /workspace/harp
COPY third_party/alpa/alpa /workspace/harp/alpa/
COPY third_party/alpa/setup.py /workspace/harp/
COPY third_party/alpa/README.md /workspace/harp/
COPY scripts /workspace/harp/scripts/
COPY benchmark /workspace/benchmark/

RUN cd /workspace/harp/ && \
    pip install -e ".[dev]" && \
    pip install -U "ray[default]"==2.9.0 && \
    pip install RainbowPrint==0.0.1 -i https://pypi.org/simple && \
    pip install numpy==1.20.0 && \
    pip install numba==0.53.0  && \
    pip install grpcio==1.60.0  && \
    pip install pydantic==1.10.13  && \
    pip install colorama && \
    pip install gin_config && \
    echo "Port 9022" >> /etc/ssh/sshd_config && \
    service ssh restart

CMD ["/usr/sbin/sshd", "-D"]

RUN cd /workspace && \
    wget https://developer.download.nvidia.cn/devtools/nsight-systems/NsightSystems-linux-cli-public-2025.6.1.190-3689520.deb && \
    dpkg -i NsightSystems-linux-cli-public-2025.6.1.190-3689520.deb && \
    rm NsightSystems-linux-cli-public-2025.6.1.190-3689520.deb

RUN pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113


# docker build \
#     --network host \
#     --build-arg http_proxy=$http_proxy \
#     --build-arg https_proxy=$https_proxy \
#     -f docker/build_harp_cu113_sm80.Dockerfile \
#     -t harp:sm80 .

# docker run -it \
#     --network host \
#     --shm-size 24G \
#     --privileged \
#     --ulimit memlock=-1 \
#     --name harp \
#     --gpus all \
#     -v /root/lssyes/nfs_share:/workspace/nfs_share \
#     harp:sm80 /bin/bash
