from flask import render_template, flash, redirect, request, url_for, abort
from flask import session as flask_session
from flask_login import current_user, login_user, logout_user, login_required, AnonymousUserMixin
from app.forms import SignupForm, LoginForm
from app import app, log, db, login_manager, models
from app.user import User, create_user
from app.session import Session
from app.util import load_img
import os
from markdown import markdown


def from_md(fname, title):
    header = f"<h3>{title}</h3>\n<hr>\n"
    if not fname.endswith(".md"):
        fname+=".md"
    path = os.path.join(app.root_path, app.static_folder, "md", fname)
    with open(path, "r") as f:
        md = header + markdown(f.read())
    return render_template("md.html", title="imgs.ai", md=md)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


'''
@app.route("/signup", methods=["GET", "POST"])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash("User name already exists; please choose another one.", "warning")
        else:
            user = create_user(form)
            flash("Thank you for requesting access, you will hear from us in the next 24 hours.", "info")
    return render_template("signup.html", title="imgs.ai", form=form)
'''


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
                session.config["private"] = True
                session.store(flask_session)
                return redirect(url_for("interface"))
            flash("Access not granted yet!", "warning")
        else:
            flash("Invalid username/password combination")
        return redirect(url_for("login"))
    return render_template("login.html", title="imgs.ai", form=form)


# TODO: Toggle design not Bootstrap 5 compatible
@app.route("/users", methods=["GET", "POST"])
@login_required
def users():
    if current_user.id==1: # Admin
        if request.method == "POST":
            for i, access in request.form.items():
                user = User.query.get(int(i))
                user.access = bool(int(access))
                db.session.commit()
        return render_template("users.html", title="imgs.ai", users=User.query.all(),)
    else:
        abort(403)


@app.route("/logs")
@login_required
def logs():
    if current_user.id==1: # Admin

        def to_md(lines):
            lines.reverse()
            md = "```\n" + "".join(lines) + "```"
            return markdown(md, extensions=['fenced_code', 'codehilite'])
        
        with(open("app/static/log.log", "r")) as f:
            return render_template("md.html", title="imgs.ai", md=to_md(f.readlines()))    
    else:
        abort(403)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flask_session.clear()
    return redirect(url_for("index"))


@app.route("/")
def index():
    return from_md("about", "About")


@app.route("/imprint")
def imprint():
    return from_md("imprint", "Imprint")


# TODO: Help page


# TODO: Move dataset descriptions to readme in dataset folder, collect from there
@app.route("/datasets_public")
def datasets_public():
    return from_md("datasets_public", "Datasets")


@app.route("/datasets_private")
@login_required
def datasets_private():
    return from_md("datasets_private", "Datasets")


# FIXME: CORS errors because servers do not always send access-control-allow-origin headers (mostly MoMA), potential solution: local dataset
@app.route("/interface", methods=["GET", "POST"])
def interface():
    session = Session(flask_session)

    if request.method == "POST":

        # Upload
        # Only present in request if filled
        if "upload" in request.files and request.files["upload"]:
            file = request.files["upload"]
            if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ["png", "jpg", "jpeg", "gif"]:
                session.edit_idxs("add_upload", upload=load_img(file))

        # Prompt
        # Only present in request if filled
        if "prompt" in request.form and request.form["prompt"]:
            session.edit_idxs("add_prompt", prompt=request.form["prompt"])

        # Idx actions
        # Only present in request if clicked
        if "action" in request.form:
            session.edit_idxs(request.form["action"], idxs=request.form.getlist("active")) # Button names are actions

        # Settings
        # Must check if different because always present in request
        if "n" in request.form:
            if request.form["n"] != session.config["n"]:
                session.edit_config("n", request.form["n"])
        if "metric" in request.form: # Must check metric first because it depends on emb_type
            if request.form["metric"] != session.config["metric"]:
                session.edit_config("metric", request.form["metric"])
        if "emb_type" in request.form:
            if request.form["emb_type"] != session.config["emb_type"]:
                session.edit_config("emb_type", request.form["emb_type"])

        # Model
        # Must check if different because always present in request
        # Must check last because overrides all other settings
        if request.form["model"] != session.config["model_name"]:
            session.load_model(request.form["model"])

    session.compute_nns()
 
    # Log search
    search_target = f'idxs +{[idx.idx for idx in session.data["pos_idxs"]]}, -{[idx.idx for idx in session.data["neg_idxs"]]}'
    # See https://github.com/mattupstate/flask-security/blob/4049c0620383f42d37950c7a35af5ddd6df0540f/flask_security/utils.py#L65
    if 'X-Forwarded-For' in request.headers:
        ip = request.headers.getlist("X-Forwarded-For")[0].rpartition(' ')[-1]
    else:
        ip = request.remote_addr or 'untrackable'
    if isinstance(current_user, AnonymousUserMixin):
        user = "Anonymous"
    else:
        user = current_user
    log.info(f'{ip} {user} searched for {search_target} in {session.config["model_name"]}, returning {len(session.data["res_idxs"])} results')

    # Store in cookie
    session.store(flask_session)

    return render_template("interface.html", title="imgs.ai", session=session)