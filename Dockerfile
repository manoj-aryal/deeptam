FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

MAINTAINER t-thanh <tien.thanh@eu4m.eu>

RUN apt-get update && apt-get install -y sudo wget
RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN export uid=1000 gid=1000
RUN mkdir -p /home/docker
RUN echo "docker:x:${uid}:${gid}:docker,,,:/home/docker:/bin/bash" >> /etc/passwd
RUN echo "docker:x:${uid}:" >> /etc/group
#RUN echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN chmod 0440 /etc/sudoers
RUN chown ${uid}:${gid} -R /home/docker

USER docker
WORKDIR /home/docker
#install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh 
RUN bash ~/Miniconda2-latest-Linux-x86_64.sh -b -p ~/miniconda2 
ENV PATH=/home/docker/miniconda2/bin:${PATH}
RUN conda update -y conda
CMD ["bash" "--gpus all -it --rm -e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME:/home/$USER"] conda init && source ~/.bashrc

RUN conda create -n deeptam python=3.5 -y && \
	conda init bash && source ~/.bashrc && conda activate deeptam && \
	pip install --upgrade pip && pip install tensorflow-gpu==1.4.0 && \
	sudo apt-get install -y build-essential cmake libboost-all-dev libeigen3-dev libx11-dev libjpeg-dev libxxf86vm1 libxxf86vm-dev libxi-dev mesa-common-dev libxext-dev libpng-dev libimlib2-dev libglew-dev libxrender-dev libxrandr-dev libglm-dev libxt-dev && \
	pip install minieigen && pip install scikit-image && \
	conda install -y cudatoolkit=8.0 cudnn=6.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/
RUN git clone -b deeptam https://github.com/lmb-freiburg/lmbspecialops.git && \
	git clone https://github.com/t-thanh/deeptam && \
	mv lmbspecialops/CMakeLists.txt lmbspecialops/CMakeLists.txt_backup && \
	cp deeptam/patch/CMakeLists.txt lmbspecialops && \
	cd lmbspecialops && mkdir build && cd build && cmake .. && make && \
	export PYTHONPATH=$PYTHONPATH:~/deeptam/tracking/python:~/deeptam/mapping/python:~/lmbspecialops/python && \
	echo $PYTHONPATH && cd ~/deeptam/tracking/data && ./download_testdata.sh && \
	cd ~/deeptam/tracking/weights && ./download_weights.sh && \
	cd ~/deeptam/mapping/weights && ./download_weights.sh 
	

ENTRYPOINT ["/bin/bash"]
