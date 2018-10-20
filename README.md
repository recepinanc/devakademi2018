# devakademi2018
dev.akademi2018 Repository

# The Project

This project uses machine learning algorithm RandomForestRegressor to predict a value I defined as `conversion rate` which is equal to the `((click_count/impression_count)*100))`, to indicate what percentage of impressions for a given ad get the users clicked to it.  
Since I have too small data to train on (188 Advertisements only) I thought ensemble models would be the best way to go, and I did, RandomForestRegressor.

Since I want to improve myself in the field of data science, I was already dedicated to do something that uses data science. The only thing I needed was an idea that combines machine learning and results in a project that actualy outputs something meaningful.

I also learned to setup a really simple Flask server for the first time in my life.

# Technologies Used

RandomForestRegressor (scikit-learn library)  
Flask - For API

# Aim

This project aims to predict the what's called `the conversion rate` for an advertisement based on its 6 main features using machine learning:
- `bid`: Bid for the ad
- `budget`: Budget for the ad
- `event_category`: Which category the ad should be shown
- `call_action_wc`: Number of words in ad_call_to_action
- `description_wc`: Number of words in ad_description
- `title_wc`: Number of words in ad_title


# How did I approach to this problem?

I first analyzed the data, to see if there are any correlations between the features in the data. After getting familiar with the dataset I started to write what comes to my mind that can be valuable either for the owner of the ads or sahibinden.com itself. After getting to know the data, I realized that it would be best if I clearly state the approach and maybe create an interface so that it can be tested. I created different models and tested them using different evaluation models, finally sticked to cross validation score, which is a great method to prevent misleading results other metrics could give due to overfitting. Finally, after building the model, I wanted to create an interface for it and I started to search for tutorials to create an API with `Flask`.

# Setting up the Environment

## Project Dependencies:

This project uses *Python2.7*, please make sure your version of python is 2.7 too:  
`python -V`

All dependencies are related to packages are stated in the `requirements.txt` file.  
To install all the packages run the following command:  
`pip install -r requirements.txt`

## Sample Usage:

To build the model from scratch:  
`python train_model.py`

This will run the *train_model.py* file which creates a `RandomForestRegressor` model and trains it using the data `devakademi.json` in the same directory.  
The output of this script will be one scaler called `min_max_scaler.sav` and a trained machine learning model called `regression_model.py`.

After it is done, start the flask server on your local at `'0.0.0.0:5000'` by running:
`python simple_flask_server.py`

This will automatically listen for your requests at `'0.0.0.0:5000/predict'`.  

This endpoint accepts data as json:
```
{
  "ad": {
          "bid": 260, 
          "budget": 4000, 
          "is_category_emlak": 0, 
          "is_category_hayvanlar_alemi": 0, 
          "is_category_vasita": 0, 
          "is_category_yedek_parca": 0, 
          "is_category_ikinci_el_ve_sifir": 1,
          "is_category_is_makineleri": 0,
          "call_to_action": "bfff8af3f7609621ceb6",
          "description": "8b36e9 ee1a793907 0ba 29c8ae3130 529230e5a4a 6e4254e 5decb4 925191d117f 6666fb1c",
          "title": "8b36e9 6f041af"
  }
}
```

Using Postman or cURL you can easily test this project by just sending a `POST` request with the payload above in the body of the request.

This will yield a response in json:
```
{
    "predictions": [
        2.6913578554995157
    ]
}
```

# What's next?

This project can be extended by adding more features after more detailed analysis of the data.  
The accuracy of the model is pretty low right now, it can easily be improved using a lot more data.
If I had more data I would try building a `Neural Network` model using `Keras`.
