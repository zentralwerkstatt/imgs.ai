conda install --yes -c pytorch -c nvidia pytorch torchvision torchaudio ${1:-cpuonly}
conda install --yes -c conda-forge flask flask-cors flask-wtf flask-login flask-migrate flask-sqlalchemy bootstrap-flask
conda install --yes requests tqdm scikit-learn h5py gunicorn dill gdown markdown email-validator
pip install annoy
pip install git+https://github.com/openai/CLIP.git
cd ../../models/public
gdown https://drive.google.com/uc?id=1XFPI_WJJk6MCZ8CT7D4UaUUJWGMEu1IP
unzip rma.zip
rm rma.zip
mv rma Rijksmuseum