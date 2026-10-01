from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']  # get input news from form
        data = [news]
        vect = vectorizer.transform(data).toarray()  # vectorize the input
        prediction = model.predict(vect)  # predict using model

        # Convert prediction to label
        if prediction[0] == 0:
            label = 'Fake'
        else:
            label = 'Real'

        # Show result on same page with label
        return render_template('index.html', prediction=label)

if __name__ == '__main__':
    app.run(debug=True)
