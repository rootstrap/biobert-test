#!/bin/bash

rm -rf BERT && git clone https://github.com/google-research/bert.git

mv bert BERT && cd BERT

wget https://github.com/naver/biobert-pretrained/releases/download/v1.1-pubmed/biobert_v1.1_pubmed.tar.gz

tar -xf biobert_v1.1_pubmed.tar.gz

rm biobert_v1.1_pubmed.tar.gz  && mv biobert_v1.1_pubmed/* . && rm -rf mv biobert_v1.1_pubmed

pip install virtualenv

virtualenv --python=/usr/bin/python3.6 venv

source venv/bin/activate

/usr/bin/python3.6 -m pip install --upgrade pip

pip3.6 install -r requirements.txt

export PATH="/home/ubuntu/.local/bin:$PATH"

source ~/.profile

cd ../ && mkidr INPUT && mkdir OUTPUT && mkdir TEMP