import pandas as pd
df=pd.read_csv("healthcare_patient_risk_dataset.csv")
print(df)

# data set information & colums check
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

# missing values & data cleaning

print(df.isnull().sum())
df=df.drop_duplicates()

print("After cleaning data:")
print(df.isnull().sum())
print("dataset shape:",df.shape)

# EDA - basic analysis

# age analysis
print("average age:",df["Age"].mean())
print("Maximum Age:",df["Age"].max())
print("Minimum Age:",df["Age"].min())

# risk level count
print("Risk  level:")
print(df["Risk_Level"].value_counts())

# gender count
print("Gender count:")
print(df["Gender"].value_counts())

# Diabetes count
print("Diabetes count:")
print(df["Diabetes"].value_counts())

# Data visualization

# Risk level graph

import matplotlib.pyplot as plt

risk_counts=df["Risk_Level"].value_counts()

plt.figure(figsize=(7,5))
plt.bar(risk_counts.index,risk_counts.values)

plt.title("Paitent risk level")
plt.xlabel("Risk level")
plt.ylabel("Number of paitant")

plt.show()


# Age vs Risklevel

risk_numeric = df["Risk_Level"].map({"Low": 0, "High": 1})

plt.figure(figsize=(8, 5))

plt.scatter(df["Age"], risk_numeric)

plt.title("Age vs Patient Risk")
plt.xlabel("Age")
plt.ylabel("Risk (0 = Low, 1 = High)")

plt.show()

# BMI VS Risk analysis

plt.figure(figsize=(8,5))
risk_numeric=df["Risk_Level"].map({"Low":0,"High":1})
plt.scatter(df["BMI"],risk_numeric)
plt.title("BMI VS Patient Risk")
plt.xlabel("BMI")
plt.ylabel("Risk (0=low,1=High)")
plt.show()

# Blood pressure vs RISK Analysis

risk_numeric=df["Risk_Level"].map({"low":0,"High":1})
plt.figure(figsize=(8,5))
plt.scatter(df["Blood_Pressure"],risk_numeric)
plt.title("Blood pressure vs Patient Risk")
plt.xlabel("Blood pressure")
plt.ylabel("Risk(0=low , 1=High)")
plt.show()


# Cholesterol vs Risk analysis

risk_numeric=df["Risk_Level"].map({"Low":0,"high":1})
plt.figure(figsize=(8,5))
plt.scatter(df["Cholesterol"],risk_numeric)
plt.title("Cholesterol vs patient risk")
plt.xlabel("Cholesterol")
plt.ylabel("Risk(0=low , 1=High)")
plt.show()

# Diabetes vs Risk analysis

diabetes_risk=pd.crosstab(df["Diabetes"],df["Risk_Level"])
print(diabetes_risk)

diabetes_risk.plot(kind="bar",figsize=(8,5))

plt.title("Diabetes vs patient risk")
plt.xlabel("Diabetes ")
plt.ylabel("Numbers of patients")
plt.xticks(rotation=0)
plt.show()

# final risk analysis

print("==Final Risk Analysis==")

print("Risk level:",df["Risk_Level"].value_counts())
print("Average by BMI by Risk:-")
print(df.groupby("Risk_Level")["BMI"].mean())
print("average by Blood pressure:-")
print(df.groupby("Risk_Level")["Blood_Pressure"].mean())
print("Average by Cholesterol:- ")
print(df.groupby("Risk_Level")["Cholesterol"].mean())
print("Average by Glucose:-")
print(df.groupby("Risk_Level")["Glucose"].mean())
print("Average by Age:-")
print(df.groupby("Risk_Level")["Age"].mean())


# FINAL CONCLUSION

print("===Project Conclusion===")
high_risk=(df["Risk_Level"]== "High").sum()
low_risk=(df["Risk_Level"]=="Low").sum()

print("Total patients:",len(df))
print("High Risk Paitent:",high_risk)
print("Low Risk Paitents:",low_risk)

