import logging
import sys
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_cors import CORS
from flask_bootstrap import Bootstrap5
from app.model import EmbeddingModel
from datetime import date
import importlib

# Logging
logging.captureWarnings(True)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(f"app/static/logs/app.log")
formatter = logging.Formatter("%(asctime)s : %(message)s", "[%d/%b/%Y:%H:%M:%S %z]")
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

config_file = os.getenv('IMGS_CONFIG')
log.info(f'Using config: {config_file}')
config_module = importlib.import_module(config_file, package=None)
Config = config_module.Config

# Start app
app = Flask(__name__)
app.config.from_object(Config)

# Plugins
Bootstrap5(app)  # Bootstrap
CORS(app)  # CORS
login_manager = LoginManager(app)  # Login
login_manager.login_view = "login"
login_manager.login_message_category = "warning"

# Database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Models
models = {}
# Pre-load both public and private models
log.info(f"Loading private models {Config.MODEL_NAMES_PRIVATE} and public models {Config.MODEL_NAMES_PUBLIC}")
for model in Config.MODEL_NAMES_PRIVATE + Config.MODEL_NAMES_PUBLIC:
    models[model] = EmbeddingModel()
    sub = "public"
    if model in Config.MODEL_NAMES_PRIVATE: sub = "private" 
    MODEL_PATH = os.path.join(Config.MODELS_PATH, sub, model)
    if os.path.isfile(os.path.join(MODEL_PATH, 'config.json')):
        models[model].load(MODEL_PATH)

from app import user, routes

# Initialize default user
if not os.path.isfile("users.db"):
    with app.app_context():
        db.create_all()
        default_user = user.User(username=Config.DEFAULT_USERNAME, email=Config.DEFAULT_EMAIL, access=True)
        default_user.set_password(Config.DEFAULT_PASSWORD)
        db.session.add(default_user)
        db.session.commit()