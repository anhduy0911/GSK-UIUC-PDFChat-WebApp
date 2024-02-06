# Use nvidia/cuda image
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
ENV TZ=America/Chicago

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
apt-get clean
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh -O ~/anaconda.sh && \
/bin/bash ~/anaconda.sh -b -p /opt/conda && \
rm ~/anaconda.sh && \
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
find /opt/conda/ -follow -type f -name '*.a' -delete && \
find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
/opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# setup conda virtual environment
COPY ./environment.yml /tmp/environment.yml
RUN conda update conda \
&& conda env create --name dna_chat -f /tmp/environment.yml

RUN echo "conda activate dna_chat" >> ~/.bashrc
ENV PATH /opt/conda/envs/dna_chat/bin:$PATH
ENV CONDA_DEFAULT_ENV $dna_chat

# actual process running
WORKDIR /chatpdf
COPY * /chatpdf/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "streamlitui.py", "--server.port=8501", "--browser.gatherUsageStats=False", "--server.address=0.0.0.0"]