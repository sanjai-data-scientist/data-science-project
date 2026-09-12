# 📈 Sales Forecasting & Business Analytics

## 📌 Project Overview
Built a data-driven sales forecasting system to help business identify key sales drivers and predict future revenue. Analyzed product, region, monthly, and salesperson-wise trends to give actionable business insights.

## 🛠️ Tools & Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **ML Models:** Linear Regression, Random Forest Regressor

## 🔍 Key Work Done
1.  **Data Cleaning:** Handled missing values, handled date formats, removed duplicates
2.  **EDA (Exploratory Data Analysis):** Analyzed sales trends by Product, Region, Month, and Salesperson
3.  **Feature Engineering:** Created new features from Date (Month, Year) for better prediction
4.  **Model Building:** 
    - Linear Regression for baseline prediction
    - Random Forest Regressor for better accuracy
5.  **Model Evaluation:** Evaluated using MAE, MSE, and R² Score
6.  **Feature Importance:** Used Random Forest to find which factors influence sales the most

## 📊 Results
- Random Forest model gave higher R² and lower error than Linear Regression
- Identified top-performing Regions, Products, and Salespersons
- Found monthly seasonality in sales

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python project01.py
