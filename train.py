import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datetime import time
import joblib
import pickle

def time_to_seconds(time):
    return time.hour * 3600 + time.minute * 60 + time.second

df = pd.read_csv('./data.csv')
df = df.drop("date", axis=1)
df = df.drop("is_holiday", axis=1)
#timestamp (int; number of seconds since beginning of day)
print(df.describe())
# print(df.columns.to_list())
# noon = time_to_seconds(time(12, 0, 0))
# df.timestamp = df.timestamp.apply(lambda t: abs(noon - t))
#print(df.describe())
#columns = ["day_of_week", "month", "hour"]
#df = pd.get_dummies(df, columns=columns)

print(df.columns.to_list())

data = df.values
X = data[:, 1:]
y = data[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#pickle.dump(scaler, open('scaler.pkl','wb'))

model = RandomForestRegressor(n_jobs=-1)

model.set_params(n_estimators=50)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
joblib.dump(model, "./rf.joblib", compress=3)

# Predict on new test set
# Columns needed: 'timestamp', 'day_of_week', 'is_weekend', 'is_holiday', 'is_start_of_semester', 'is_during_semester', 'month', 'hour'

# Testing for Sanity Check that the Model Works
time = time(1, 0, 0)
timestamp = time_to_seconds(time)
# timestamp = abs(noon - timestamp)
day_of_week = 6
is_weekend = 1
startsem = 0
schoolsem = 1
month = 9
temperature = 70
hour = time.hour

data = {'timestamp': timestamp, 'day_of_week': day_of_week,
    'is_weekend': is_weekend, 'temperature': temperature, 'is_start_of_semester': startsem, 
    'is_during_semester': schoolsem, 'month':month, 'hour':hour}

topred = pd.DataFrame(data, index=[0])
print(topred)
topred = scaler.transform(topred)
print(topred)
predicted = model.predict(topred)
print(predicted)
print(round(predicted[0], 0))
