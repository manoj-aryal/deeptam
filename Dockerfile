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
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
RUN bash ~/Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3 
ENV PATH=/home/docker/miniconda3/bin:${PATH}
RUN conda update -y conda
CMD ["bash" "--gpus all -it --rm -e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME:/home/$USER"] conda init && source ~/.bashrc
RUN sudo apt-get install -y build-essential cmake libboost-all-dev libeigen3-dev libx11-dev libjpeg-dev libxxf86vm1 libxxf86vm-dev libxi-dev mesa-common-dev libxext-dev libpng-dev libimlib2-dev libglew-dev libxrender-dev libxrandr-dev libglm-dev libxt-dev git

RUN git clone https://github.com/manoj-aryal/deeptam && \
	git clone -b deeptam https://github.com/lmb-freiburg/lmbspecialops.git && \
	mv lmbspecialops/CMakeLists.txt lmbspecialops/CMakeLists.txt_backup && \
	cp deeptam/patch/CMakeLists.txt lmbspecialops && \
	cd ~/deeptam/tracking/weights && ./download_weights.sh && \
	cd ~/deeptam/mapping/weights && ./download_weights.sh && \
	cd ~/deeptam/tracking/data && ./download_testdata.sh 

SHELL ["/bin/bash", "-c"]
RUN conda create -n deeptam python=3.5 -y
RUN echo "conda activate deeptam" > ~/.bashrc
ENV PATH /home/docker/miniconda3/envs/deeptam/bin:$PATH
RUN /bin/bash -c "echo `python --version` && pip install --upgrade pip && pip install tensorflow-gpu==1.4.0 && \
	pip install minieigen && pip install scikit-image && \
	conda install -y cudatoolkit=8.0 cudnn=6.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/"
ENV PYTHONPATH $PYTHONPATH:/home/docker/deeptam/tracking/python:/home/docker/deeptam/mapping/python:/home/docker/lmbspecialops/python
ENV LD_LIBRARY_PATH /home/docker/miniconda3/envs/deeptam/lib:/home/docker/miniconda3/lib/:$LD_LIBRARY_PATH
RUN chmod +x /home/docker/deeptam/install_test.sh
# Launch
ENTRYPOINT ["/home/docker/deeptam/install_test.sh"]
CMD tail -f /dev/null
