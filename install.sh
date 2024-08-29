conda install --yes -c pytorch -c nvidia pytorch torchvision torchaudio ${1:-cpuonly}
conda install --yes -c conda-forge flask flask-cors flask-wtf flask-login flask-migrate flask-sqlalchemy email-validator nodejs
conda install --yes requests tqdm scikit-learn h5py gunicorn dill gdown
pip install annoy bootstrap-flask torchray umap-learn
pip install git+https://github.com/openai/CLIP.git
cd app/static
npm install
cd ../../models/public
gdown https://drive.google.com/uc?id=1jVEjmKwv0wwTX2vt7WzAYF8F59_Tsg5s
unzip rma.zip
rm rma.zip
mv rma Rijksmuseum