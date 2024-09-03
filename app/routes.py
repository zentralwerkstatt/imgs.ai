from flask import render_template, flash, redirect, request, url_for, Response
from flask import session as flask_session
from flask_login import current_user, login_user, logout_user, login_required
from app.forms import SignupForm, LoginForm
from app import app, log, db, login_manager, models
from app.user import User, create_user
from app.session import Session
from app import Config
import time
import os
from markdown import markdown


def from_md(fname):
    if not fname.endswith(".md"):
        fname+=".md"
    path = os.path.join(app.root_path, app.static_folder, "md", fname)
    with open(path, "r") as f:
        md = markdown(f.read())
    return render_template("md.html", title="imgs.ai", Config=Config, md=md)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


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
    flash("Please note: if you signed up before September 2024 your login data has expired and you will be required to sign up again. Please check back after October 2024 for additional instructions.", "info")
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


@app.route("/")
def index():
    return from_md("about")


@app.route("/imprint")
def imprint():
    return from_md("imprint")


@app.route("/datasets_public")
def datasets_public():
    return from_md("datasets_public")


@app.route("/datasets_private")
@login_required
def datasets_private():
    return from_md("datasets_private")
    

# TODO: implement image lightbox with Bootstrap modal, see https://getbootstrap.com/docs/5.0/components/modal/
@app.route("/full/<idx>")
def full(idx):
    session = Session(flask_session)
    # FIXME: if previous action was selection, coming back from redirect attempts to POST and triggers "form resubmission message"
    return redirect(session.get_url(idx), code=302)


@app.route("/source/<idx>")
def source(idx):
    session = Session(flask_session)
    # Metadata is guaranteed for all non-uploaded files
    # See make_model in train.py
    if not idx.startswith("upload") and session.get_metadata(idx)[1]: # Has filled URL field        
        return redirect(session.get_metadata(idx)[1], code=302)
    else:
        flash("No source available for selected image", 'info')
        return render_template("interface.html", title="imgs.ai", session=session, Config=Config)


# TODO: better controls UI
# TODO: help page
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
    # FIXME: don't just remove CLIP prompt as that looks like undefined behavior on the interface
    # TODO: CLIP prompts become "images" that can be kept, removed, or turned negative
    session.clip_prompt = ""
    search_target = f"idxs +{session.pos_idxs}, -{session.neg_idxs}"
    if session.emb_type.startswith("clip"):
        if "clip_prompt" in request.form and request.form["clip_prompt"]:
            session.clip_prompt = request.form["clip_prompt"]
            search_target = f"'{session.clip_prompt}'"
            
    # Get NNs
    start = time.process_time()
    session.get_nns()


    # Logging
    # See https://github.com/mattupstate/flask-security/blob/4049c0620383f42d37950c7a35af5ddd6df0540f/flask_security/utils.py#L65
    if 'X-Forwarded-For' in request.headers:
        remote_addr = request.headers.getlist("X-Forwarded-For")[0].rpartition(' ')[-1]
    else:
        remote_addr = request.remote_addr or 'untrackable'

    log.info(
        f"Search by {remote_addr} for {search_target} in {session.model} completed in {time.process_time() - start}, returning {len(session.res_idxs)} results"
    )

    # Store in cookie
    session.store(flask_session)

    return render_template("interface.html", title="imgs.ai", session=session, Config=Config)