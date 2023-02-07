from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, SelectField
from wtforms.validators import DataRequired, URL
from flask_ckeditor import CKEditorField
from flask_wtf.file import FileField, FileRequired


class CreatePostForm(FlaskForm):
    title = StringField("Proj Title", validators=[DataRequired()])
    subtitle = StringField("Subtitle", validators=[DataRequired()])
    img_url = StringField("Blog Image URL", validators=[DataRequired(), URL()])
    body = CKEditorField("md", validators=[DataRequired()])
    category = SelectField('Category', choices=[(
        'x', 'CS50X'), ('a', 'CS50AI'), ('d', '100Day with python'), ('c', 'Chemistry')])
    submit = SubmitField("Submit Post")


class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField(label='password', validators=[DataRequired()])
    submit = SubmitField(label='Register')


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField(label='password', validators=[DataRequired()])
    submit = SubmitField(label='login')


class PdfForm(FlaskForm):
    pdff_ = FileField(validators=[FileRequired()])
    #page = SelectField('Page', choices=[(
    #    1, '1'), (2, '2'), (3, '3'), (5, '5'), (0, '0, for more subscrib for a premium to access all page stealing')])
    submit = SubmitField("Submit file")
