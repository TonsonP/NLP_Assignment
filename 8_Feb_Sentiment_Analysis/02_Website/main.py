from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from LSTM_predict import *
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mykey'
app.config['UPLOAD_FOLDER'] = 'static/files' 

@app.route('/')
def index():
    return redirect(url_for('sentiment'))


####Reddit API Part
# reddit crawler
import pandas as pd
import praw

reddit = praw.Reddit(client_id='', 
                     client_secret='', 
                     user_agent='',
                     check_for_async=False)

class MyForm(FlaskForm):
    name = StringField('Insert your topic', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/sentiment', methods = ['GET','POST'])
def sentiment():
    name = False
    results = 0,0
    str_results = 'Nope'
    form = MyForm()
    print(form.validate_on_submit())
    if form.validate_on_submit():
        name = form.name.data 
        topics = get_post(name)
        results = sentence_checking(topics)
        str_results = check_range(results)
    return render_template("sentiment.html",form=form, name=name, results=results, str_results=str_results)

if __name__ == "__main__":
    app.run(debug=True)