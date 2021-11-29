from app.util import save_imgs_to, CLIP_gradcam
from flask import render_template, flash, redirect, request, url_for, send_from_directory, Response
from flask import session as flask_session
from flask_login import current_user, login_user, logout_user, login_required
from app.forms import SignupForm, LoginForm
from app import app, log, db, login_manager, models
from app.user import User, create_user
from app.session import Session
from app import Config
import time
import os
from functools import lru_cache


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route("/")
def index():
    return redirect(url_for("interface"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash("User name already exists; please choose another one.", 'warning')
        else:
            user = create_user(form)
            flash("Thank you for requesting access, you will hear from us in the next 24 hours.", 'info')
    return render_template("signup.html", title="imgs.ai", Config=Config, form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    session = Session(flask_session)
    if current_user.is_authenticated:
        return redirect(url_for("interface"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(password=form.password.data):
            if user.access:
                log.info(f"{user.username} logged in")
                login_user(user)
                session.public = False
                session.store(flask_session)
                return redirect(url_for("interface"))
            flash("Access not granted yet!")
        else:
            flash("Invalid username/password combination")
        return redirect(url_for("login"))
    return render_template("login.html", title="imgs.ai", Config=Config, form=form)


@app.route("/users", methods=["GET", "POST"])
@login_required
def users():
    if request.method == "POST":
        for i, access in request.form.items():
            user = User.query.get(int(i))
            user.access = bool(int(access))
            db.session.commit()
    return render_template("users.html", title="imgs.ai", Config=Config, users=User.query.all(),)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flask_session.clear()
    return redirect(url_for("index"))


@app.route("/cdn/<idx>")
@lru_cache(maxsize=1000)
def cdn(idx):
    session = Session(flask_session)
    root, path, _ = session.get_data(idx)
    return send_from_directory(root, path)


@app.route("/full/<idx>")
def full(idx):
    session = Session(flask_session)
    return redirect(session.get_url(idx), code=302)


@app.route("/source/<idx>")
def source(idx):
    session = Session(flask_session)
    _, _, metadata = session.get_data(idx)
    url = metadata[1]
    return redirect(url, code=302)


@app.route("/explain/<idx>")
def explain(idx):
    session = Session(flask_session)
    url = session.get_url(idx)
    abs_path = save_imgs_to([url], Config.CACHE)[0][0] # Returns two lists
    hm = CLIP_gradcam(session.clip_prompt, abs_path).convert("RGB")
    hm.save(os.path.join(Config.CACHE, f"{idx}_hm.jpg"))
    hm_url = url_for('cdn', idx=f"{idx}_hm")
    return redirect(hm_url, code=302)


@app.route("/interface", methods=["GET", "POST"])
def interface():
    # Load from cookie
    session = Session(flask_session)

    # Uploads
    if request.files:
        session.extend(request.files.getlist("file"))

    # Settings
    if "n" in request.form:
        session.n = request.form["n"]
    if "emb_type" in request.form:
        session.emb_type = request.form["emb_type"]
        session.metrics = models[session.model].config["emb_types"][session.emb_type]["metrics"]
    if "metric" in request.form:
        session.metric = request.form["metric"] if request.form["metric"] in session.metrics else session.metrics[0]
    if "size" in request.form:
        session.size = request.form["size"]

    # Actions
    if "btn" in request.form:
        if request.form["btn"] == "Positive":
            new_pos = set(request.form.getlist("add-pos"))
            session.pos_idxs = list(set(session.pos_idxs) | new_pos) # Union of sets
            session.neg_idxs = list(set(session.neg_idxs) - new_pos)  # Difference of sets

        elif request.form["btn"] == "Remove":
            removables = set(request.form.getlist("remove"))
            session.pos_idxs = list(set(session.pos_idxs) - removables)  # Difference of sets
            session.neg_idxs = list(set(session.neg_idxs) - removables)  # Difference of sets

        elif request.form["btn"] == "Negative":
            session.neg_idxs = list(set(session.neg_idxs) | set(request.form.getlist("add-neg")))  # Union of sets
            session.pos_idxs = list(set(session.pos_idxs) - set(request.form.getlist("add-neg")))  # Difference of sets

        elif request.form["btn"] == "Clear":
            session.neg_idxs = []
            session.pos_idxs = []
            session.res_idxs = []

    if 'add-pos' in request.form:
        new_pos = set(request.form.getlist("add-pos"))
        session.pos_idxs = list(set(session.pos_idxs) | new_pos) # Union of sets
        session.neg_idxs = list(set(session.neg_idxs) - set(request.form.getlist("add-pos")))  # Difference of sets

    if 'add-neg' in request.form:
        session.neg_idxs = list(set(session.neg_idxs) | set(request.form.getlist("add-neg")))  # Union of sets
        session.pos_idxs = list(set(session.pos_idxs) - set(request.form.getlist("add-neg")))  # Difference of sets

    # Model
    if "model" in request.form:
        if session.model != request.form["model"]: # Only reload and reset if model changed
            session.load_model(request.form["model"], pin_idxs=session.pos_idxs) # Keep all positive queries

    # Determine if CLIP functions necessary
    session.clip_prompt = ""
    search_target = f"idxs +{session.pos_idxs}, -{session.neg_idxs}"
    if session.emb_type.startswith("clip"):
        if "clip_prompt" in request.form and request.form["clip_prompt"]:
            session.clip_prompt = request.form["clip_prompt"]
            search_target = f"'{session.clip_prompt}'"
            
    # Get NNs
    start = time.process_time()
    session.get_nns()
    log.info(
        f"Search by {request.remote_addr} for {search_target} in {session.model} completed in {time.process_time() - start}, returning {len(session.res_idxs)} results"
    )

    # Store in cookie
    session.store(flask_session)

    return render_template(
        "interface.html",
        title="imgs.ai",
        session=session,
        Config=Config,
    )