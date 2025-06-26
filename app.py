from flask import Flask, render_template, request
import joblib
from utils.preprocess import clean_text

app = Flask(__name__)
model = joblib.load('model/sentiment_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = ''
    if request.method == 'POST':
        review = request.form['review']
        cleaned = clean_text(review)
        sentiment = model.predict([cleaned])[0]
        ajkfhafh = print(sentiment)
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
