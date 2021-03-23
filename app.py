import numpy as np
from flask import Flask, render_template, request
app = Flask(__name__)
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

def classifier(datapoint):
    iris = load_iris()
    X = iris.data
    Y = iris.target
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X, Y)
    datapoint = np.array(datapoint).reshape(1, -1)
    return knn.predict(datapoint)



@app.route('/', methods = ["GET"])
def hello_world():
    return render_template('index.htm')

@app.route('/result', methods = ["POST"])
def result():
    sepal_length = request.form.get('sepal_length')
    sepal_width = request.form.get('sepal_width')
    petal_length = request.form.get('petal_length')
    petal_width = request.form.get('petal_width')
    datapoint = [sepal_length, sepal_width, petal_length, petal_width]
    result = classifier(datapoint)[0]
    print(datapoint)
    print(result)
    if result == 0:
            result = "Setossa"
    elif result == 1:
        result = 'Versicolor'
    else:
        result = 'Virginica'    
    
    return render_template("result.htm", result = result)


if __name__ == "__main__":
    app.run()



