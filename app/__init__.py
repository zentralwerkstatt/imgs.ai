import logging
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_cors import CORS
from flask_bootstrap import Bootstrap5
from app.model import EmbeddingModel
from app.jar import Jar


# Logging
logging.captureWarnings(True)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(f"app/static/log.log")
formatter = logging.Formatter("%(asctime)s : %(message)s", "[%d/%b/%Y:%H:%M:%S %z]")
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

# Start app
app = Flask(__name__)

# Config
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or os.urandom(32)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:////Users/fabian/Desktop/imgs.ai/users.db" # Absolute
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_COOKIE_SECURE"] = False # Activate in production
app.config["REMEMBER_COOKIE_SECURE"] = False # Activate in production
app.config["DEFAULT_USERNAME"] = "hi@imgs.ai" # Change in production
app.config["DEFAULT_EMAIL"] = "hi@imgs.ai" # Change in production
app.config["DEFAULT_PASSWORD"] = "hi@imgs.ai" # Change in production

# Plugins
Bootstrap5(app)  # Bootstrap
CORS(app)  # CORS
login_manager = LoginManager(app)  # Login
login_manager.login_view = "login"
login_manager.login_message_category = "warning"

# Database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Pre-load both public and private models
models = {}
for model in next(os.walk("app/static/models"))[1]:
    log.info(f"Loading model {model}")
    models[model] = EmbeddingModel(f"app/static/models/{model}")

# "Redis"
jar = Jar("app/static/user_content/sessions")

from app import user, routes

# Initialize default user
if not os.path.isfile("users.db"):
    with app.app_context():
        db.create_all()
        default_user = user.User(username=app.config["DEFAULT_USERNAME"], email=app.config["DEFAULT_EMAIL"], access=True)
        default_user.set_password(app.config["DEFAULT_PASSWORD"])
        db.session.add(default_user)
        db.session.commit()