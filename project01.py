import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("sales_data.csv")
print(df)

print(df.head())
print(df.shape)
print(df.info())

# =====DATA Cleaning====
print(df.isnull().sum())

print("duplicate rows:",df.duplicated().sum())

df=df.drop_duplicates()
print("removeing duplicate values:",df.shape)

df["Region"]=df["Region"].fillna(df["Region"].mode()[0])
df["Salesperson"]=df["Salesperson"].fillna(df["Salesperson"].mode()[0])

df["Date"]=pd.to_datetime(df["Date"])
print("dste",df.dtypes)

print(df.isnull().sum())


# setp 3 EDA (Exploratory data analysis)
# ********Total sales*********

total_sales=df["Sales"].sum()
print("Total sales :",total_sales)

#********** Total average ***********

average_sales=df["Sales"].mean()
print("Average_sales:",average_sales)

# ******Total quantity sold*********

total_quantity=df["Quantity"].sum()
print("Total_Quantity Sold :",total_quantity)

# ^^^^^^^^^Product-wise sales^^^^^^^^^^^^
product_sales=df.groupby("Product")["Sales"].sum()
print(product_sales)

# ^^^^Region - wise sales
region_sales=df.groupby("Region")["Sales"].sum()
print(region_sales)

# ^^^^^^^^^Sales person wise sales^^^^^^^^^^^
salesperson_sales=df.groupby("Salesperson")["Sales"].sum()
print(salesperson_sales)

# ^^^^^^^^^^^^Top sales persons^^^^^^^^^^^^^^^^^
top_salesperson=salesperson_sales.idxmax()
top_sales=salesperson_sales.max()
print("Top sales person:",top_salesperson)
print("Top sales :",top_sales)


# ************DATA VISUALIZATION****************
# Price -wise sales bar chart
product_sales=df.groupby("Product")["Sales"].sum()
plt.figure(figsize=(10,5))
product_sales.plot(kind="bar")

plt.xlabel("PRODUCT")
plt.ylabel("TOP SALES ")
plt.title("Sales by product")
plt.xticks(rotation=45)
plt.show()

# Region -wise sales
region_sales=df.groupby("Region")["Sales"].sum()
plt.figure(figsize=(10,5))
region_sales.plot(kind="bar")

plt.title("Sales by region")
plt.xlabel("Region")
plt.ylabel("Total sales")
plt.xticks(rotation=45)
plt.show()

# Monthly sales - line chart
df['Month']=df["Date"].dt.to_period("M")

Monthly_sales=df.groupby("Month")["Sales"].sum()

plt.figure(figsize=(10,5))
Monthly_sales.plot(kind="line",marker="o",color="red")

plt.title("Monthly sales trend")
plt.xlabel("Month")
plt.ylabel("total sales")
plt.xticks(rotation=45)
plt.grid()
plt.show()

salesperson_sales=df.groupby("Salesperson")["Sales"].sum()

plt.figure(figsize=(10,5))
salesperson_sales.plot(kind="bar",lw=2)

plt.xlabel("Sales person")
plt.ylabel("Total sales")
plt.xticks(rotation=45)
plt.title("SALES BY SALESPERSON")
plt.show()

# ^^^^^^^^^^^^^^^^^feature engineering^^^^^^^^^^^^^^^^
# new colums created
# date -->year
df["Year"]=df["Date"].dt.year

# Date ---> Month
df["Month"]=df["Date"].dt.month

# Date --> Day

df["Day"]=df["Date"].dt.day

# revenue created
df["Revenue"]=df["Quantity"] * df["Price"]

df["Sales_category"]=pd.cut(df["Sales"],
bins=[0,50000,100000,float("inf")],
labels=["Low","Medium","High"])

print(df.head())

# MACHINE LEARNING PREDICT SALES

x=df[["Quantity","Price","Month","Year"]]
y=df["Sales"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(y_pred[:10])

result=pd.DataFrame({
    "Actual sales":y_test.values,
    "Predicted sales":y_pred
})

print(result.head(10))

#MODEL EVELUTION 

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("Model evelution")
print(">>MAE:",mae)
print(">>MSE:",mse)
print(">>R2:",r2)


plt.figure(figsize=(12,6))
plt.plot(y_test.values,label="Actual sales ",marker="o")
plt.plot(y_pred,label="Predicted sales",marker="x")
plt.xlabel("test data")
plt.ylabel("sales")
plt.title("Actual vs predict")
plt.legend()
plt.grid()
plt.show()

result=pd.DataFrame({
    "actual sales ":y_test.values,
    "predicted sales":y_pred
})

print(result.head(10))


from sklearn.ensemble import RandomForestRegressor

rf_model=RandomForestRegressor(n_estimators=100,random_state=42)

rf_model.fit(x_train,y_train)

rf_pred=rf_model.predict(x_test)
print(rf_pred[:10])
rf_r2=r2_score(y_test,rf_pred)
print("rf_r2",rf_r2)
rf_mae=mean_absolute_error(y_test,rf_pred)
print("RF_MAE:",rf_mae)


print("linear regression r2:",r2)
print("RANDON FOREST R2:",rf_r2)

print("LinearRegression  MAE:",mae)
print("RandomForest MAE:",rf_mae)


# NEW PRODUCT PREDICT
new_data=pd.DataFrame({
    "Quantity":[5],
    "Price":[25000],
    "Month":[9],
    "Year":[2026]
}
)

prrdict=rf_model.predict(new_data)
print("Predict sales :",prrdict[0])


predicted_sales=rf_model.predict(new_data)[0]

print("===SALES PREDICTION===")
print("Quantity:",new_data["Quantity"][0])
print("Price:",new_data["Price"][0])
print("Month:",new_data["Month"][0])
print("Year:",new_data["Year"][0])
print("-----------------------------")
print(f"Predicted_sales:₹{predicted_sales:,.2f}")


importance=rf_model.feature_importances_
feature=x.columns

feature_importance=pd.DataFrame({
    "Features":feature,
    "Importance":importance
})

print(feature_importance)

plt.figure(figsize=(8,5))

plt.bar(
    feature_importance["Features"],
    feature_importance["Importance"]
)

plt.title("features importance")
plt.xlabel("features")
plt.ylabel("importances")
plt.show()
