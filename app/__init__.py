from flask import Flask, render_template, request
from ML import cv, model

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        user = request.form.get('text')
        data = cv.transform([user]).toarray()
        output = model.predict(data)
        prediction = output[0]
        print(output)

    except(ValueError):
        print('Seems you have entered a value that is not in our database, you can try again')
        #The program is only 0.9531 accuracy of getting the language so there are bound to be misconceptions.

    return render_template('index.html', prediction= 'That phrase is in: {}'.format(prediction))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)