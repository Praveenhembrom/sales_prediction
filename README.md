# Sales Prediction Project 

##  Project Overview
This project focuses on building a **machine learning model** to forecast **weekly sales** for Walmart stores.  
The motivation is to help retailers **understand demand, optimize inventory, and improve business decisions**.  

We followed a **step-by-step data science workflow**:
1. Integrating and cleaning data from multiple sources.  
2. Exploring trends and patterns using visualization.  
3. Engineering meaningful features to capture seasonality and store-level dynamics.  
4. Training and tuning machine learning models.  
5. Generating predictions and saving them in the required format.  

The final solution uses **LightGBM**, a gradient boosting algorithm well-suited for structured/tabular data, combined with **lag and rolling features** to model sales trends over time.

---

##  Workflow of the Project

### **Stage 1: Setup and Data Integration**
- Imported essential libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`).
- Loaded datasets: `train.csv`, `test.csv`, `stores.csv`, `features.csv`.
- Merged data into **train_df** and **test_df** on `Store` and `Date`.
- Converted `Date` column into proper datetime format.

---

### **Stage 2: Exploratory Data Analysis (EDA)**
- Checked dataset size, data types, and missing values.  
- Filled missing values with **median imputation** to reduce bias.  
- Visualized key patterns:
  - Sales trends across weeks and months.  
  - Store-wise total performance (which stores sell the most).  
  - Impact of store type and size on sales.  
  - Effect of holiday weeks on sales compared to normal weeks.  

---

### **Stage 3: Feature Engineering**
- Extracted **time-based features**: `Year`, `Month`, `WeekOfYear`, `Day`.  
- Applied **one-hot encoding** to categorical features (`Type`).  
- Dropped redundant or noisy features (`Date`, `MarkDown` columns).  
- Added **advanced time-series features**:  
  - `Sales_Lag_1` → Previous week’s sales.  
  - `Sales_Roll_Mean_4` → Rolling 4-week average sales.  

---

### **Stage 4: Model Preparation**
- Defined **X (features)** and **y (target: Weekly_Sales)**.  
- Used **time-based split** (80% training, 20% validation) instead of random split to reflect real-world forecasting.  

---

### **Stage 5: Model Training & Initial Evaluation**
- Chose **LightGBM Regressor** for training.  
- Trained on `X_train, y_train` and validated on `X_val, y_val`.  
- Evaluated with:
  - **MAE (Mean Absolute Error)** → Easy to interpret average error.  
  - **RMSE (Root Mean Square Error)** → Penalizes large deviations.  

---

### **Stage 6: Model Optimization**
- Used **Optuna** for hyperparameter tuning.  
- Tested multiple parameter combinations automatically.  
- Selected the **best-performing set** and retrained the model for final use.  

---

### **Stage 7: Final Prediction and Saving**
- Generated predictions for the **test dataset**.  
- Formatted output into `submission.csv` for competition submission.  
- Saved the trained model with `joblib` for future use.  

---


