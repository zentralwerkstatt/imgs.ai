conda install --yes -c pytorch pytorch torchvision torchaudio $1
conda install --yes -c conda-forge flask flask-cors flask-wtf flask-login flask-migrate flask-sqlalchemy email-validator nodejs
conda install --yes requests tqdm scikit-learn h5py gunicorn cmake
pip install pybase64 annoy face_recognition bootstrap-flask
pip install git+https://github.com/openai/CLIP.git
cd app/static
npm install