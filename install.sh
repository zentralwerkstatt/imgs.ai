conda create --yes -n imgs.ai python=3.8
source activate imgs.ai
conda install --yes -c pytorch pytorch torchvision torchaudio $1
conda install --yes -c conda-forge flask flask-cors flask-wtf flask-login flask-migrate flask-sqlalchemy email-validator nodejs
conda install --yes requests tqdm scikit-learn h5py gunicorn cmake
pip install pybase64 annoy face_recognition bootstrap-flask
cd app/static
npm install