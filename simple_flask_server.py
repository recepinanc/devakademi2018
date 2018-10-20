import flask
import pickle
import numpy as np

# Load Flask, Scaler and Model
app = flask.Flask(__name__)
scaler = pickle.load(open("min_max_scaler.sav","rb"))
model = pickle.load(open("regression_model.pkl","rb"))

# define a predict function as an endpoint
@app.route("/predict", methods=["GET","POST"])
def predict():
    if 'feature_array' in flask.request.get_json().keys() :
        feature_array = flask.request.get_json()['feature_array']
        print("Received Features:", feature_array)

        # Scale the features
        scaled = scaler.transform(np.array(feature_array).reshape(1, -1))
        print("Scaled Features:", scaled)
        
        prediction = model.predict(scaled).tolist()
        print("Predicted the Conversion Rate %:", prediction)
        #preparing a response object and storing the model's predictions
        response = {}
        response['predictions'] = prediction
    else:
        response = {'result': 'failure due to missing feature set'}
        
    # response
    return flask.jsonify(response)

# start the flask app, allow remote connections
app.run(host='0.0.0.0')
