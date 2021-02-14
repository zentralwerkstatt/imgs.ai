from flask import render_template, flash, redirect, request, url_for, send_from_directory, get_flashed_messages, Response
from flask import session as flask_session
from flask_login import current_user, login_user, logout_user, login_required
from app.forms import SignupForm, LoginForm, EmbedderForm
from app import app, log, db, login_manager, models
from model import EmbeddingModel
from app.user import User, create_user
from app.session import Session
from app import Config
import time
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from functools import lru_cache


def cond_login_required(function):
    if Config.USER_MGMT:
        return login_required(function)
    else:
        return function


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route("/")
def index():
    return render_template("index.html", title="imgs.ai", Config=Config)


@app.route("/help")
@cond_login_required
def help():
    return render_template("help.html", title="imgs.ai", Config=Config)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash("User name already exists; please choose another one.", 'warning')
        else:
            user = create_user(form)
            if user.access:
                login_user(user)  # Log in as newly created user
                return redirect(f"{url_for('interface')}")
            flash("Thank you for requesting beta access, you will hear from us in the next 24 hours.", 'info')
    return render_template("signup.html", title="imgs.ai - Sign up for alpha", Config=Config, form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("interface"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(password=form.password.data):
            if user.access:
                log.info(f"{user.username} logged in")
                login_user(user)
                return redirect(url_for("interface"))
            flash("Access not granted yet!")
        else:
            flash("Invalid username/password combination")
        return redirect(url_for("login"))
    return render_template("login.html", title="Log in", Config=Config, form=form)


@app.route("/users", methods=["GET", "POST"])
@cond_login_required
def users():
    if request.method == "POST":
        for i, access in request.form.items():
            user = User.query.get(int(i))
            user.access = bool(int(access))
            db.session.commit()
    return render_template("users.html", title="Users", Config=Config, users=User.query.all(),)


@app.route("/logout")
@cond_login_required
def logout():
    logout_user()
    flask_session.clear()
    return redirect(url_for("index"))


@app.route("/settings")
@cond_login_required
def settings():
    session = Session(flask_session)
    return render_template("settings.html", title="imgs.ai", Config=Config, session=session)


@app.route("/cdn/<idx>")
@cond_login_required
@lru_cache(maxsize=100)
def cdn(idx):
    session = Session(flask_session)
    root, path, _, _ = session.get_data(idx)
    return send_from_directory(root, path)


@app.route("/interface", methods=["GET", "POST"])
@cond_login_required
def interface():
    # Load from cookie
    session = Session(flask_session)

    # Uploads
    if request.files:
        session.extend(request.files.getlist("file"), Config.UPLOADS_PATH)

    # Settings
    if "n" in request.form:
        session.n = request.form["n"]
    if "emb_type" in request.form:
        session.emb_type = request.form["emb_type"]
    if "metric" in request.form:
        session.metric = request.form["metric"]
    if "mode" in request.form:
        session.mode = request.form["mode"]
    if "size" in request.form:
        session.size = request.form["size"]

    # Actions
    if "btn" in request.form:
        if request.form["btn"] == "Positive":
            new_pos = set(request.form.getlist("add-pos"))
            session.pos_idxs = list(set(session.pos_idxs) | new_pos) # Union of sets
            session.neg_idxs = list(set(session.neg_idxs) - new_pos)  # Difference of sets
            log.debug(f'{current_user} added {len(new_pos)} positives')

        elif request.form["btn"] == "Remove":
            removables = set(request.form.getlist("remove"))
            session.pos_idxs = list(set(session.pos_idxs) - removables)  # Difference of sets
            session.neg_idxs = list(set(session.neg_idxs) - removables)  # Difference of sets
            log.debug(f'{current_user} removed {removables} from search')

        elif request.form["btn"] == "Negative":
            session.neg_idxs = list(set(session.neg_idxs) | set(request.form.getlist("add-neg")))  # Union of sets
            session.pos_idxs = list(set(session.pos_idxs) - set(request.form.getlist("add-neg")))  # Difference of sets

        elif request.form["btn"] == "Clear":
            session.neg_idxs = []
            session.pos_idxs = []

        elif request.form["btn"] == "Export":
            # https://flask.palletsprojects.com/en/1.1.x/patterns/streaming/
            def generate():
                for idx in session.pos_idxs:
                    _, path, _, _ = session.get_data(idx)
                    yield '+,' + path + '\n'
                for idx in session.neg_idxs:
                    _, path, _, _ = session.get_data(idx)
                    yield '-,' + path + '\n'
            return Response(generate(), mimetype='text')

    if 'add-pos' in request.form:
        new_pos = set(request.form.getlist("add-pos"))
        session.pos_idxs = list(set(session.pos_idxs) | new_pos) # Union of sets
        session.neg_idxs = list(set(session.neg_idxs) - set(request.form.getlist("add-pos")))  # Difference of sets
        log.debug(f'{current_user} added {len(new_pos)} positives')

    if 'add-neg' in request.form:
        session.neg_idxs = list(set(session.neg_idxs) | set(request.form.getlist("add-neg")))  # Union of sets
        session.pos_idxs = list(set(session.pos_idxs) - set(request.form.getlist("add-neg")))  # Difference of sets

    # Model
    if "model" in request.form:
        if session.model != request.form["model"]: # Only reload and reset if model changed
            session.load_model(request.form["model"], pin_idxs=session.pos_idxs) # Keep all positive queries

    start = time.process_time()
    # CLIP search
    if "clip_prompt" in request.form and request.form["clip_prompt"] and session.emb_type == "clip":
        session.get_nns_CLIP(request.form["clip_prompt"])
    # Regular search
    else:
        session.get_nns()
    log.info(
        f"Search by {current_user} in {session.model} completed in {time.process_time() - start}, returning {len(session.res_idxs)} results"
    )

    # Render data
    popovers, links, images = session.render_nns()

    # Store in cookie
    session.store(flask_session)

    log.debug(f'Positive indices: {session.pos_idxs}')
    log.debug(f'Negative indices: {session.neg_idxs}')

    return render_template(
        "interface.html",
        title="imgs.ai",
        session=session,
        Config=Config,
        popovers=popovers,
        links=links,
        images=images
    )