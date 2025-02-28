#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install numpy')
import numpy as np 
n_users = 1000  
click_probability = 0.3  
clicks = np.random.binomial(1, click_probability, n_users) 
# Step 2: Calculate Click-Through Rate (CTR)
total_clicks = np.sum(clicks)  # Total number of clicks
ctr = total_clicks / n_users  
print(f"Total Users: {n_users}")
print(f"Total Clicks: {total_clicks}")
print(f"Click-Through Rate (CTR): {ctr:.2%}")


# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)


time_of_day = np.random.randint(1, 24, 500) 
day_of_week = np.random.randint(1, 6, 500)  
cars_per_minute = np.random.poisson(lam=3, size=500)  

# Increase Bank  during peak hours (8 AM to 9 AM, 5 PM to 7 PM)
for i in range(len(cars_per_minute)):
    if (time_of_day[i] >= 8 and time_of_day[i] <= 9) or (time_of_day[i] >= 17 and time_of_day[i] <= 19):
        cars_per_minute[i] = np.random.poisson(lam=8)  

data = {
    'time_of_day': time_of_day,
    'day_of_week': day_of_week,
    'cars_per_minute': cars_per_minute
}

df = pd.DataFrame(data)

X = df[['time_of_day', 'day_of_week']]
y = df['cars_per_minute']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = PoissonRegressor(alpha=1e-12)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

print("\nPredictions on test data:")
for i, pred in enumerate(y_pred):
    print(f"Sample {i+1}: Predicted cars per minute = {pred:.0f}, Actual = {y_test.iloc[i]}")

# Confusion matrix
y_pred_rounded = np.round(y_pred)
cm = confusion_matrix(y_test, y_pred_rounded)

print("\nConfusion Matrix:\n", cm)


# In[ ]:





# In[ ]:




