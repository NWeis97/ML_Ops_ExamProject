# Base image
FROM python:3.9-slim

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

# Copy files to docker
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY configs/ configs/

# Install ML_things and google.cloud storage
RUN apt update
RUN apt install -y git
RUN apt install -y wget
RUN rm -rf /var/lib/apt/lists/*
RUN pip3 install git+https://github.com/gmihaila/ml_things.git
RUN pip3 install google-cloud-storage

# Install requirements
WORKDIR /
RUN pip3 install -r requirements.txt --no-cache-dir

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip3 install cloudml-hypertune


# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Define entrypoint for docker image
# -u: redirects any print statements to our consol
# Otherwise find print statements in docker log
ENTRYPOINT ["python3", "-u", "src/models/train_model.py"]





