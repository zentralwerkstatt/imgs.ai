import os
import pathlib


class Config(object):
    SECRET_KEY = os.environ.get("SECRET_KEY") or os.urandom(32)
    print('Secret key:', SECRET_KEY)

    current_dir = pathlib.Path(__file__).parent.absolute()

    SQLALCHEMY_DATABASE_URI = f"sqlite:///{current_dir}/users.db"  # Absolute
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    UPLOADS_PATH = f"{current_dir}/uploads"  # Absolute
    MODELS_PATH = f"{current_dir}/models"  # Absolute
    MODELS = [f.name for f in os.scandir(MODELS_PATH) if f.is_dir()]
    
    NS = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
    DEFAULT_N = "30"
    SIZES = ["32", "64", "96", "128", "160", "192", "224"]
    DEFAULT_SIZE = "128"
    MODES = ["ranking", "centroid"]
    DEFAULT_MODE = "ranking"

    SESSION_COOKIE_SECURE = True # Activate in production
    REMEMBER_COOKIE_SECURE = False # Activate in production
    USER_MGMT = False # Activate in production
    TRAINING = True # Deactivate in production