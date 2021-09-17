pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
pip install mmdet
cd classifyText/mmocr1
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
export PYTHONPATH=$(pwd):$PYTHONPATH
cd ../..

cd classifyText/pan/post_processing/
rm -rf pse.so
make
pip install polygon3
pip install pyclipper
pip install colorlog
python -m pip install Pillow==6.2
cd ../../..

# gdown --id 1-U38UAigrcEgzKEZZgjtF-kHnbZfEYbv
# unzip checkpoints.zip
# rm -f checkpoints.zip

pip install pyspellchecker
pip install ngram==3.3.2
pip install -U scikit-learn
conda install -c anaconda tensorflow-gpu==2.4.1
pip install pandas
pip install tqdm
pip install seaborn
pip install natsort
