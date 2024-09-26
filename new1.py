#Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = {
    'Area (sq ft)': [1500, 2500, 1800, 2200, 1400, 2600, 3000, 1750, 2800, 2100,1400, 2600, 3000, 2500, 1800, 2200],
    'Bedrooms': [3, 4, 3, 4, 2, 5, 4, 3, 4, 3, 2, 5, 4,4, 3, 4],
    'Bathrooms': [2, 3, 2, 3, 1, 4, 3, 2, 3, 2, 1, 4, 3,1, 4, 3],
    'Year_Built': [2000, 2010, 2005, 2012, 1995, 2015, 2020, 2003, 2018, 2008, 2015, 2020, 2003,2020, 2003, 2018],
    'Parking':["N","Y","N","Y","N","Y","Y","N","Y","Y","N","Y","Y","Y","N","Y"], # Yes/No
    'Property_Type':["U","S","U","S","U","S","F","U","S","S","U","S","F","S","U","S",],  # SemiFurnished/Unfurnsihed/Furnished
    }

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)
# Add the 'target' column (Price ($))
df['target'] = [3000, 5000, 3600, 4400, 2800, 5200, 6000, 3600, 5600, 4200,2800, 5200, 6000,5000, 3600, 4400]
# Display the DataFrame
print("df = ","\n",df)

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
print("\----\n","step-1:")
print(df.info())

#convert categorical data into numeric
print("\----\n","step-2:")
categorical_data = df.select_dtypes(exclude=[np.number])
print("---\n categorical_data=\n\n",categorical_data)
categorical_data_features = df.select_dtypes(exclude=[np.number]).columns
print("---\n categorical_data_features=\n\n",categorical_data_features)
df[categorical_data_features] = df[categorical_data_features].apply(lambda x:pd.factorize(x)[0])
print("df = ","\n",df)

# create heatmap for data visualisation
print("\----\n","step-3:")
plt.figure(figsize=(12,8))
sns.heatmap(df.drop(columns='target').corr(),annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Heatmap of feature correlation')
plt.savefig('heatmap_corelation.png')
plt.pause(2)# shows the graph for 2 sec only
plt.close()
print("\n","Heat Map Created")

# segreagate the dataframe in features and target
print("\----\n","step-4:")
X = df.drop(columns='target')
print("\----\n","X:",X)
Y = df['target']
print("\----\n","Y:",Y)
#SCALING
print("\----\n","step-5:")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\----\n","X_scaled:",X_scaled)
print("\----\n","X_scaled.shape:",X_scaled.shape)

# slit into training and test
print("\----\n","step-6:")
X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,Y,test_size = 0.2,random_state = 42)
print("\n","X.shape=",X.shape)
print("\n","X_train.shape=",X_train.shape)
print("\n","X_test.shape=",X_test.shape)

print("\n","Y.shape=",Y.shape)#SINGLE COLUMN VECTOR
print("\n","Y_train.shape=",Y_train.shape)
print("\n","Y_test.shape=",Y_test.shape)

# TRAIN lINER rEGRESSION mODEL
print("\----\n","step-7:")
model = LinearRegression()
model.fit(X_train,Y_train)
print("\n","Model has been trained")
#print coefficients and slope
print("\----\n","step-8:")
print("\n","Coefficient(slope):",model.coef_)
print("\n","Itercept:",model.intercept_)

# 9 predict values ontraining data
print("\----\n","step-9:")
Y_train_pred = model.predict(X_train)
print("\n","Y_train_pred:",Y_train_pred)
# r squared value
train_r2 = r2_score(Y_train,Y_train_pred)
print("\----\n","step-10:")
print("\n","R-squaredvalue for training data:",train_r2)
#predict values
print("\----\n","step-11:")
Y_test_pred = model.predict(X_test)
print("\n","Y_test_predicted",Y_test_pred)

#rsquared on test data
print("\n\n","Step-12:")
test_r2=r2_score(Y_test,Y_test_pred)
print("\n","R-squared value for test data:",test_r2)

print("\n\n","Step-13:")
result_df=pd.DataFrame({'Y_test':Y_test.values,'Y_test_pred':Y_test_pred})
result_df.to_csv('Y_test_vs_Y_test_pred.csv',index=False)
print("CSV file created.")

#visualize data
print("\n\n","Step-14:")
plt.figure(figsize=(10,6))
plt.plot(Y_test.values,label='Actual Prices(Y_test)',color='blue')
plt.plot(Y_test_pred,label='Predicted Prices(Y_test_pred)',color='red',linestyle='--')
plt.title('Actual vs Predicted Prices(Test Data)')
plt.xlabel('Data Points')
plt.ylabel('Prices')
plt.legend()
plt.savefig('Test_Pred_Actual.png')
plt.show()
