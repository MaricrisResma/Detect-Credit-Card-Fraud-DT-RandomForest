import pickle
from flask import Flask, render_template, request

# OOps concept
# Class, Objects, Methods, Inheritance, Polymorphism, Abstraction, Encapsulation, Decorators
# Generators, Dunder Methods, Abstract methods, Static Methods

#Create an object of the class Flask


app = Flask(__name__)

#undump model
model = pickle.load(open('model.pkl','rb'))

# url /
@app.route('/')
def index():
    return render_template('index.html')

# endpoint
@app.route('/predit', methods=['GET', 'POST'])
def predict():
    # prediction = model.predict([[28]])
    prediction = model.predict([[request.form.get('temperature')]])
    output = round(prediction[0],2)
    print(prediction)
    #return None
    return render_template('index.html', prediction_text=f'Total revenue generated is CAD. {output}/-')


if __name__=='__main__':
    app.run(debug=True)