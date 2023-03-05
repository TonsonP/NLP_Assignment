from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired


# Manually create vocabulary.
# Used defaultdict as it can return the values for the unknown key.
from collections import defaultdict

def def_w2i_val():
    return 0
def def_i2w_val():
    return '<unk>'

### LOAD OBJECTS ###
word2index_th = defaultdict(def_w2i_val)
word2index_en = defaultdict(def_w2i_val)
index2word_th = defaultdict(def_i2w_val)
index2word_en = defaultdict(def_i2w_val)

import pickle
file = open('/root/projects/NLP/Assignment/23_Feb_Machine_translation/object/sm_word2index_en.pkl', 'rb')
word2index_en = pickle.load(file)

file = open('/root/projects/NLP/Assignment/23_Feb_Machine_translation/object/sm_word2index_th.pkl', 'rb')
word2index_th = pickle.load(file)

file = open('/root/projects/NLP/Assignment/23_Feb_Machine_translation/object/sm_index2word_th.pkl', 'rb')
index2word_th = pickle.load(file)

file = open('/root/projects/NLP/Assignment/23_Feb_Machine_translation/object/sm_index2word_en.pkl', 'rb')
index2word_en = pickle.load(file)
from translation_file import *
### END OF LOAD OBJECTS ###


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mykey'
app.config['UPLOAD_FOLDER'] = 'static/files' 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@app.route('/')
def index():
    return redirect(url_for('machinetranslation'))


class MyForm(FlaskForm):
    name = StringField('Type something', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/machinetranslation', methods = ['GET','POST'])
def machinetranslation():
    if request.method == 'POST':
        source = request.form.get('source')
        predict = translating(source)
    else:
        source = ' '
        predict = ' '

    data = {"source":source, "predict":predict}
    return render_template("machinetranslation.html", data = data)

if __name__ == "__main__":
    app.run(debug=True)