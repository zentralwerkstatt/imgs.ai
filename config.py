import os
import pathlib

class Config(object):
    SECRET_KEY = os.environ.get("SECRET_KEY") or os.urandom(32)

    current_dir = pathlib.Path(__file__).parent.absolute() # TODO: os.path alternative
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{current_dir}/users.db"  # Absolute
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MODELS_PATH = f"{current_dir}/models"  # Absolute, storage for models
    DATA_PATH = f"{current_dir}/app/static/data" # Data storage for local models and uploads, symlink if necessary
    MODEL_NAMES_PRIVATE = sorted([f.name for f in os.scandir(os.path.join(MODELS_PATH, "private")) if f.is_dir()])
    MODEL_NAMES_PUBLIC = sorted([f.name for f in os.scandir(os.path.join(MODELS_PATH, "public")) if f.is_dir()])

    NS = ["25", "50", "75", "100"]
    DEFAULT_N = "50"
    SIZES = ["32", "64", "128", "256"]
    DEFAULT_SIZE = "128"

    SESSION_COOKIE_SECURE = False # Activate in production
    REMEMBER_COOKIE_SECURE = False # Activate in production

    DEFAULT_USERNAME = "hi@imgs.ai" # Change in production
    DEFAULT_EMAIL = "hi@imgs.ai" # Change in production
    DEFAULT_PASSWORD = "hi@imgs.ai" # Change in production