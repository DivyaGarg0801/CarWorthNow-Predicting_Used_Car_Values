import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

df = pd.read_csv('used_cars_data.csv')
# Split the cleaned 'car_name' column into 'Car_Company' and 'Model_Name'
df[['Car_Company', 'Model_Name','Car_specs']] = df['Name'].str.split(' ', n=2, expand=True)
# Drop the 'car_name' column as it's no longer needed
df = df.drop(columns=['Name','Car_specs'])
columns = ['Car_Company', 'Model_Name'] + [col for col in df.columns if col not in ['Car_Company', 'Model_Name']] # Get the list of current columns
df = df[columns] # Reorder the DataFrame


label_encoder = LabelEncoder()
encoders = {}
for column in ['Car_Company', 'Model_Name', 'Location', 'Fuel_Type', 'Transmission', 'Ownership']:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])  # Transform the training data
    encoders[column] = encoder  # Save the encoder for reuse

df['Manufacturing_Year'] = pd.to_numeric(df['Manufacturing_Year'], errors='coerce')
q1=df['Manufacturing_Year'].quantile(q=0.25)
q3=df['Manufacturing_Year'].quantile(q=0.75)
iqr=q3-q1
iqr_ll=q1-1.5*iqr
iqr_ul=q3
df = df[(df["Manufacturing_Year"].isnull()) |((df['Manufacturing_Year'] >= iqr_ll) & (df['Manufacturing_Year'] <= iqr_ul))]


#df.boxplot(column=["Fuel_Type","Seats","Kms_Driven","Ownership","Transmission","Mileage (kmpl)","Engine (cc)","Max_Power (bhp)"])
#plt.show()

df['Max_Power (bhp)'] = df['Max_Power (bhp)'].apply(lambda x: ''.join(char for char in str(x) if char.isdigit()))
df['Max_Power (bhp)'] = pd.to_numeric(df['Max_Power (bhp)'], errors='coerce')
q3=df['Max_Power (bhp)'].quantile(q=0.75)
q1=df['Max_Power (bhp)'].quantile(q=0.25)
iqr=q3-q1
iqr_ll=q1-1.5*iqr
iqr_ul=q3+1.5*iqr
df = df[(df["Max_Power (bhp)"].isnull()) |((df['Max_Power (bhp)'] >= iqr_ll) & (df['Max_Power (bhp)'] <= iqr_ul))]
#df.boxplot(column=['Max_Power (bhp)'])
#plt.show()

df['Engine (cc)'] = df['Engine (cc)'].apply(lambda x: ''.join(char for char in str(x) if char.isdigit()))
df['Engine (cc)'] = pd.to_numeric(df['Engine (cc)'], errors='coerce')
q3=df['Engine (cc)'].quantile(q=0.75)
q1=df['Engine (cc)'].quantile(q=0.25)
iqr=q3-q1
iqr_ll=q1-1.5*iqr
iqr_ul=q3+0.5*iqr
df = df[(df["Engine (cc)"].isnull()) |((df['Engine (cc)'] >= iqr_ll) & (df['Engine (cc)'] <= iqr_ul))]
#df.boxplot(column=['Engine (cc)'])
#plt.show()

q3=df['Seats'].quantile(q=0.75)
q1=df['Seats'].quantile(q=0.25)
iqr=q3-q1
iqr_ll=q1-1.5*iqr
iqr_ul=q3+1.5*iqr
df = df[(df["Seats"].isnull()) |((df['Seats'] >= iqr_ll) & (df['Seats'] <= iqr_ul))]
#df.boxplot(column=['Seats'])
#plt.show()

df['Mileage (kmpl)'] = df['Mileage (kmpl)'].apply(lambda x: ''.join(char for char in str(x) if char.isdigit()))
df['Mileage (kmpl)'] = pd.to_numeric(df['Mileage (kmpl)'], errors='coerce')
q3=df['Mileage (kmpl)'].quantile(q=0.75)
q1=df['Mileage (kmpl)'].quantile(q=0.25)
iqr=q3-q1
iqr_ll=q1-1.5*iqr
iqr_ul=q3+1.5*iqr
df = df[(df["Mileage (kmpl)"].isnull()) |((df['Mileage (kmpl)'] >= iqr_ll) & (df['Mileage (kmpl)'] <= iqr_ul))]
#df.boxplot(column=['Mileage (kmpl)'])
#plt.show()

q3=df['Kms_Driven'].quantile(q=0.75)
q1=df['Kms_Driven'].quantile(q=0.25)
iqr=q3-q1
iqr_ll=q1-1.5*iqr
iqr_ul=q3+1.2*iqr
df = df[(df["Kms_Driven"].isnull()) |((df['Kms_Driven'] >= iqr_ll) & (df['Kms_Driven'] <= iqr_ul))]
#df.boxplot(column=['Kms_Driven'])
#plt.show()


q3=df['Price (in lakhs)'].quantile(q=0.9)
q1=df['Price (in lakhs)'].quantile(q=0.25)
iqr=q3-q1
iqr_ll=q1-1.5*iqr
iqr_ul=q3+1.5*iqr
df = df[(df['Price (in lakhs)'] >= iqr_ll) & (df['Price (in lakhs)'] <= iqr_ul)]
df.boxplot(column=['Price (in lakhs)'])
plt.show()

#df.boxplot(column=["Fuel_Type","Seats","Kms_Driven","Ownership","Transmission","Mileage (kmpl)","Engine (cc)","Max_Power (bhp)",'Price (in lakhs)'])
#plt.show()

df['Manufacturing_Year'] = df['Manufacturing_Year'].astype(str).where(df['Manufacturing_Year'].astype(str).str.isdigit(), '')
df['Mileage (kmpl)'] = df['Mileage (kmpl)'].fillna(df['Mileage (kmpl)'].median())
df["Engine (cc)"] = df["Engine (cc)"].fillna((df["Engine (cc)"]).median())
df["Max_Power (bhp)"]=df["Max_Power (bhp)"].fillna((df["Max_Power (bhp)"]).mean())

print("Modified Dataframe:")
print(df)

a=df.isnull().sum()
print(a)

print(df.shape)
df.dropna(inplace=True)
print(df.shape)

df.drop_duplicates(inplace=True)
print(df.shape)


input_data = df[["Car_Company","Model_Name","Location","Manufacturing_Year","Fuel_Type","Seats","Kms_Driven","Ownership","Transmission","Mileage (kmpl)","Engine (cc)","Max_Power (bhp)"]]
output_data=df["Price (in lakhs)"]
x_train,x_test,y_train,y_test=train_test_split(input_data,output_data,test_size=0.3,random_state=0)
#random_state=0: Ensures the same random split is generated every time you run the code for reproducibility.
# Feature scaling (optional but recommended for better performance)


model = RandomForestRegressor()
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(f"Random Forest R-squared accuracy: {accuracy * 100:.2f}%")

# Make predictions
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse=mse**0.5
print(f"Mean Squared Error: {mse}")

pickle.dump(model,open('model.pkl','wb'))


with open('encoders.pkl','wb') as f:
    pickle.dump(encoders,f)

