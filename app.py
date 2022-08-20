import numpy as np
from flask import Flask, request, render_template
from model import main


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    playlist_link = request.form['playlist link']
    output=main(playlist_link)
    my_songs = []
    for i in range(10):
      my_songs.append([str(output.iloc[i,2]) + ' - '+ '"'+str(output.iloc[i,4])+'"', "https://open.spotify.com/track/"+ str(output.iloc[i,0]).split(":")[-1]])

    return render_template('index.html', songs=my_songs)


if __name__ == "__main__":
    app.run(debug=True)