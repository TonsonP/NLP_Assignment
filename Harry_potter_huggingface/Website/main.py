from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import pandas as pd
#from predict_auto import predict
#from text_generator import *
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mykey'
app.config['UPLOAD_FOLDER'] = 'static/files' 

import torch
from transformers import pipeline

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "right" # "left" or "right"
tokenizer.pad_token = tokenizer.eos_token

@app.route('/')
def index():
    return redirect(url_for('autocomplete'))


class MyForm(FlaskForm):
    name = StringField('Type something', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/autocomplete', methods = ['GET','POST'])
def autocomplete():
    form = MyForm()
    code = False
    name = False
    print(form.validate_on_submit())
    if form.validate_on_submit():
        name = form.name.data 
        # code = predict(prompt = name, temperature=0.5)
        pipe = pipeline("text-generation", max_length=100, pad_token_id=0, eos_token_id=0, model='TonsonP/Harry_potter_story_generator', tokenizer=tokenizer)
        code = pipe(name, num_return_sequences=50)[0]["generated_text"]
        #code = generate_sentence(name)
        form.name.data = ""
    return render_template("autocomplete.html",form=form,name =name, code=code)

if __name__ == "__main__":
    app.run(debug=True)