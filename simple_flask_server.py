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
    if 'ad' in flask.request.get_json().keys() :
        #feature_array = flask.request.get_json()['feature_array']
        ad_params = {
                        "bid": 0, 
                        "budget": 0, 
                        "is_category_emlak": 0, 
                        "is_category_hayvanlar_alemi": 0, 
                        "is_category_vasita": 0, 
                        "is_category_yedek_parca": 0, 
                        "is_category_ikinci_el_ve_sifir": 0,
                        "is_category_is_makineleri": 0,
                        "call_to_action": "",
                        "description": "",
                        "title": ""
                    }

        ad = flask.request.get_json()['ad']

        # Parse Advertisement
        bid = ad['bid']
        budget = ad['budget']
        is_category_emlak = ad['is_category_emlak']
        is_category_hayvanlar_alemi = ad['is_category_hayvanlar_alemi']
        is_category_vasita = ad['is_category_vasita']
        is_category_yedek_parca = ad['is_category_yedek_parca']
        is_category_ikinci_el_ve_sifir = ad['is_category_ikinci_el_ve_sifir']
        is_category_is_makineleri = ad['is_category_is_makineleri']
        call_to_action_wc = len(ad['call_to_action'].split())
        description_wc = len(ad['description'].split())
        title_wc = len(ad['title'].split())

        feature_array = [bid, budget, is_category_emlak, is_category_hayvanlar_alemi, is_category_vasita, is_category_yedek_parca, is_category_ikinci_el_ve_sifir, is_category_is_makineleri, call_to_action_wc, description_wc, title_wc]

        print("Advertisement:", ad)
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
