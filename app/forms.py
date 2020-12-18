from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField, TextAreaField, FileField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional


# https://hackersandslackers.com/flask-login-user-authentication/
class SignupForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField(
        "Email",
        validators=[
            Length(min=6),
            Email(message="Enter a valid email."),
            DataRequired(),
        ],
    )
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Length(min=6, message="Select a stronger password."),
        ],
    )
    confirm = PasswordField(
        "Confirm Your Password",
        validators=[
            DataRequired(),
            EqualTo("password", message="Passwords must match."),
        ],
    )
    submit = SubmitField("Register", render_kw={"class": "btn btn-light"})


class LoginForm(FlaskForm):
    email = StringField(
        "Email", validators=[DataRequired(), Email(message="Enter a valid email.")]
    )
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Log In", render_kw={"class": "btn btn-light"})


class EmbedderForm(FlaskForm):
    projectName = StringField('Project name', validators=[DataRequired()])
    urlPerLineFile = FileField('CSV file', validators=[DataRequired()], render_kw={'accept': '.csv'})
    submit = SubmitField("Train", render_kw={"class": "btn btn-light", "onclick": "submitActive()"})