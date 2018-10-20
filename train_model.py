import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

dataset = pd.read_json('dev_akademi.json')
dataset = dataset[['ad_bid_price_kurus','ad_call_to_action', 'ad_daily_budget_kurus', 'ad_title', 'ad_id', 'ad_description', 'event_type', 'event_category']]

# Data Preprocessing

df = dataset.dropna()
df = df[df.ad_call_to_action != ""]
df = df[df.event_type != ""]
df = df[df.event_category != ""]

# Feature Engineering
modify_df = pd.concat([df, pd.get_dummies(df.event_type, prefix='event')], axis=1)
modify_df = pd.concat([modify_df, pd.get_dummies(df.event_category, prefix='event_cat')], axis=1)

# word count
call_to_action_wc = modify_df.ad_call_to_action.apply(lambda x: len(x.split()))
description_wc = modify_df.ad_description.apply(lambda x: len(x.split()))
title_wc = modify_df.ad_title.apply(lambda x: len(x.split()))
modify_df = modify_df.assign(call_to_action_wc=call_to_action_wc)
modify_df = modify_df.assign(description_wc=description_wc)
modify_df = modify_df.assign(title_wc=title_wc)

cleaned_df = modify_df.drop(['ad_title', 'ad_description', 'event_type', 'event_category'], axis=1)
grouped_df = cleaned_df.groupby('ad_id')['event_CLICK', 'event_IMPRESSION'].sum()

# Conversion Rates
conversion_rates = ((grouped_df.event_CLICK/grouped_df.event_IMPRESSION)*100).reset_index()
conversion_rates.columns = ['ad_id', 'rate']
conversion_rates = conversion_rates.set_index('ad_id')

indexed_df = cleaned_df.set_index('ad_id')
joined_df = indexed_df.join(conversion_rates)

to_scale_df = joined_df.drop(['ad_call_to_action', 'event_CLICK', 'event_IMPRESSION'], axis=1)
to_scale_df = to_scale_df.drop_duplicates()

# Feature Scaling

scaler = MinMaxScaler()
X = to_scale_df.drop(['rate'], axis=1)
y = to_scale_df.rate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train = scaler.fit_transform(X_train)

print("Scaled attributes.")

pickle.dump(scaler, open('min_max_scaler.sav', 'wb'))

print("Scaler saved.")

# GridSearch to Fine-tune model

param_grid = [
    {'n_estimators': [3, 10], 'max_features': [2, 4, 'auto']}
]

random_regressor = RandomForestRegressor()
grid_search = GridSearchCV(random_regressor, param_grid, cv=5,
                           scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# View results

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Pick the best model
best_regressor = grid_search.best_estimator_
best_regressor.n_jobs = 1
best_regressor.fit(X_train, y_train)
print("Random Forest Model (Fine Tuned) is Trained")


# Evaluation
scaled_test_set = scaler.transform(X_test)
y_pred = best_regressor.predict(scaled_test_set)

print("MAE: ", mean_absolute_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))
print("Cross Validation Score (cv=10): ", cross_val_score(best_regressor, scaled_test_set, y_test, cv=10))

#serializing our model to a file called model.pkl
pickle.dump(best_regressor, open("regression_model.pkl","wb"))

#loading a model from a file called model.pkl
model = pickle.load(open("regression_model.pkl","rb"))


print("Trained and saved model")
