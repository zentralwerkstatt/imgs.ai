from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, Config


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    access = db.Column(db.Boolean())

    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method="scrypt")

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"{self.username} ({self.email})"


def create_user(form):
    user = User(username=form.name.data, email=form.email.data, access=False)
    user.set_password(form.password.data)
    db.session.add(user)
    db.session.commit()
    return user
