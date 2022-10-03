from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    with open("predictions.csv") as f:
        return render_template('home.html', csv=f)

if __name__=="__main__":
    app.run(debug=True)