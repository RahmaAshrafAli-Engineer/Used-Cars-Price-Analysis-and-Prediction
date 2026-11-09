# 🚗 Used Cars Market Analysis & Price Prediction

<div align="center">

**From Raw Data to Intelligent Price Predictions**

An End-to-End Data Science Project combining Exploratory Data Analysis, Feature Engineering, Machine Learning, Model Explainability, and Interactive Deployment.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas)
![Scikit Learn](https://img.shields.io/badge/Scikit--learn-Machine_Learning-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Modeling-EC6B23)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-8A2BE2)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## 📌 Project Overview

The used car market is influenced by multiple interconnected factors, making price estimation more complex than simply looking at a vehicle's age or mileage.

This project explores the US used car market through a complete data science workflow, aiming to understand the key drivers behind vehicle prices and develop a reliable machine learning model for price prediction.

Rather than treating prediction as a black-box problem, the project combines statistical exploration, data-driven insights, model evaluation, and explainability to understand not only **how much a car is worth, but also why.**

The project covers the complete journey from raw data to an interactive prediction application.

## 🎯 Project Objectives

- Understand the characteristics and distribution of the used car market.
- Identify the most influential factors affecting vehicle prices.
- Investigate relationships between vehicle age, mileage, brand, condition, and selling price.
- Prepare high-quality data through systematic cleaning and feature engineering.
- Develop and compare multiple regression models.
- Evaluate model performance using appropriate regression metrics.
- Interpret model predictions using SHAP Explainability.
- Deploy the trained model through an interactive Streamlit application.

## 📊 Dataset Overview

The dataset contains **5,500 used car records and 18 original features**, covering different aspects of vehicles, including:

| Category | Features |
|---|---|
| Vehicle Identity | Make, Model, Year |
| Technical Specifications | Engine Size, Horsepower, Torque |
| Usage | Mileage, Owners |
| Vehicle Condition | Accident History, Service History |
| Market Information | Selling Price, Location |
| Vehicle Classification | Fuel Type, Transmission, Body Type, Drivetrain |
| Additional Attributes | Color, Fuel Efficiency |

After preprocessing and feature engineering, the dataset was expanded to **25 columns**.

## 🔍 Project Workflow

### Phase 1 — Data Understanding & Quality Assessment

Before making any changes, the dataset was examined to understand its structure and identify potential quality issues.

Key activities:
- Dataset structure and data type inspection.
- Missing-value distribution analysis.
- Duplicate detection.
- Statistical summaries and distribution analysis.
- Identifying potential data quality issues.

**Important observation:** Missing values were distributed across the dataset rather than concentrated in specific vehicle categories, making a thoughtful imputation strategy more appropriate than deleting records.

### Phase 2 — Data Cleaning & Preprocessing

Data cleaning was performed with the goal of preserving useful information while improving data consistency.

Key techniques:
- Group-based imputation using the median of `Model` and `Fuel_Type`.
- Handling categorical missing values without making unsupported assumptions.
- Investigating outliers before deciding whether to retain them.
- Preserving genuine extreme observations.
- Identifying the $500 price floor affecting approximately 22% of records.

Instead of automatically removing outliers, their context was investigated to distinguish unusual but valid observations from potential errors.

### Phase 3 — Exploratory Data Analysis (EDA)

A detailed exploratory analysis was conducted to understand the relationships between vehicle characteristics and selling prices.

Areas explored:

- Price distribution and skewness.
- Brand and model price comparisons.
- Fuel type and vehicle category comparisons.
- Correlation analysis.
- Vehicle age and depreciation patterns.
- Mileage and usage intensity.
- Accident history and its relationship with vehicle age.
- Electric and hybrid vehicle price comparisons.
- Depreciation patterns across different brands.

**Key Analytical Insights**

| Finding | Result |
|---|---|
| Age vs. Price | Strong negative correlation of -0.80 |
| Mileage vs. Price | Strong negative relationship |
| Annual Depreciation | Approximately 14% |
| Accident Price Penalty | Approximately 18–26% within comparable age groups |
| Electric Cars | Higher overall median prices, largely explained by younger vehicle age |
| Brand Depreciation | Approximately 14.4–16.2% annually |

One important finding was that comparing accident-damaged and accident-free cars without considering vehicle age can produce misleading conclusions.

After comparing cars within similar age groups, the estimated accident-related price penalty became substantially more realistic.

### Phase 4 — Feature Engineering

New features were introduced to capture additional information and improve the modeling process.

| Feature | Description |
|---|---|
| `Age` | Vehicle age derived from manufacturing year |
| `Log_Price` | Log-transformed selling price |
| `Mileage_per_Year` | Average mileage relative to vehicle age |
| `Is_Luxury` | Identifies selected luxury brands |
| `Price_Floor` | Identifies vehicles affected by the $500 minimum price |

Feature engineering helped represent vehicle usage patterns and brand positioning more effectively.

### Phase 5 — Machine Learning Model Development

The modeling phase focused on predicting vehicle selling prices using regression algorithms.

**Target Variable:** `Log_Price`

A logarithmic transformation was applied to reduce price skewness and support model learning. Predictions were converted back to dollar values for evaluation.

**Data Preparation**
- Train/Test Split: 80% Training / 20% Testing.
- Training Samples: 4,400.
- Testing Samples: 1,100.
- Random State: 42.
- Stratification based on `Price_Floor`.

**Preprocessing Pipeline**
- One-Hot Encoding for categorical variables.
- Standard Scaling for numerical variables.
- ColumnTransformer for structured preprocessing.
- Scikit-learn Pipelines to organize preprocessing and model training.

**Models Developed**

1. Linear Regression — Baseline Model.
2. Random Forest Regressor.
3. XGBoost Regressor.

All models were evaluated using the same test set to ensure a fair comparison.

### 📈 Model Evaluation & Performance

Multiple regression metrics were used to evaluate model performance:

- **R²:** Measures how well the model explains price variation.
- **MAE:** Measures the average absolute prediction error.
- **RMSE:** Penalizes larger prediction errors more heavily.

**Performance Comparison**

| Model | R² (Log) | R² ($) | MAE ($) | RMSE ($) |
|---|---:|---:|---:|---:|
| Linear Regression | 0.919 | 0.736 | 3,046 | 6,609 |
| Random Forest | 0.971 | 0.962 | 1,542 | 2,502 |
| **XGBoost** | **0.984** | **0.984** | **913** | **1,639** |

<details>
<summary><b>🏆 Best Model: XGBoost</b></summary>

XGBoost achieved the strongest performance among the evaluated models.

- R²: 0.984
- MAE: approximately $913
- RMSE: approximately $1,639

The MAE was approximately 70% lower than the Linear Regression baseline.

</details>

**Overfitting & Cross-Validation Analysis**

To examine model generalization, training performance was compared against testing performance, followed by 5-fold cross-validation.

| Evaluation | Result |
|---|---:|
| Training R² | 0.996 |
| Testing R² | 0.984 |
| 5-Fold CV Mean R² | 0.984 |
| Cross-Validation Std | 0.001 |

The small performance gap and consistent cross-validation results indicate stable performance across the evaluated folds.

### 🧠 Phase 6 — Model Explainability with SHAP

High predictive performance alone does not explain how a model reaches its decisions.

SHAP (SHapley Additive exPlanations) was used to interpret the selected XGBoost model and investigate the contribution of different features to its predictions.

**Feature Impact Distribution**

| Feature | Share of Total Impact |
|---|---:|
| Age | 25.4% |
| Model | 23.9% |
| Mileage | 15.6% |
| Is_Luxury | 9.8% |
| Service_History | 6.9% |
| Accident_History | 5.0% |

Explainability techniques included:

- SHAP Summary Plot.
- Grouped Feature Importance.
- Individual Prediction Waterfall Plot.

The explainability results reinforced several findings from the exploratory analysis: vehicle age, model, and mileage were among the most influential price drivers.

This provided an additional layer of validation between the exploratory findings and the model's learned behavior.

### 🌐 Phase 7 — Interactive Deployment with Streamlit

The trained model was integrated into an interactive Streamlit application to make price predictions accessible through a user-friendly interface.

The application connects the trained machine learning model with a practical user experience, allowing users to interact with the prediction system without manually executing notebook code.

**Application Preview**

<!-- Add your Streamlit screenshots here -->

*Screenshots of the application will be added here.*

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation and numerical computing |
| Matplotlib & Seaborn | Statistical visualization |
| Plotly & Altair | Interactive data visualization |
| Scikit-learn | Preprocessing, pipelines, and evaluation |
| Random Forest | Ensemble regression |
| XGBoost | Gradient boosting regression |
| SHAP | Model explainability |
| Joblib | Model serialization |
| Streamlit | Interactive application development |

## 📁 Repository Structure

```text
Used-Cars-Price-Analysis-and-Prediction/
│
├── data/
│   ├── automobile_dataset.csv
│   └── cleaned_cars.csv
│
├── notebooks/
│   ├── 01_Data_Analysis_EDA.ipynb
│   └── 02_Model_Development.ipynb
│
├── app/
│   ├── app.py
│   └── model.joblib
│
├── presentation/
│   └── project_presentation.pdf
│
├── images/
│   └── application_screenshots/
│
├── requirements.txt
├── .gitignore
└── README.md
```

*The structure above represents the intended organization of the repository.*

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Used-Cars-Price-Analysis-and-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Used-Cars-Price-Analysis-and-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app/app.py
```

*Make sure the dataset and trained model are placed in their expected locations before running the project.*

## 📌 Key Takeaways

This project demonstrates the importance of combining analytical thinking with machine learning development.

Beyond achieving strong predictive performance, the process highlighted several important lessons:

- Data quality and preprocessing decisions directly influence model reliability.
- Correlation does not necessarily imply causation.
- Understanding confounding variables is essential for drawing meaningful conclusions.
- Feature engineering can provide additional predictive information.
- Model evaluation should go beyond a single performance metric.
- Explainability is an important part of building trustworthy machine learning solutions.

## 🔮 Future Improvements

- Experiment with additional regression algorithms.
- Perform hyperparameter optimization.
- Investigate model performance across different price segments.
- Improve prediction uncertainty estimation.
- Expand application functionality and deployment options.
- Evaluate generalization on independent real-world datasets.

## 👩‍💻 Author

**Rahma Ashraf Ali Faragallah**

Data Science & AI Enthusiast | Machine Learning Developer

Interested in transforming complex datasets into meaningful insights and developing intelligent, practical solutions.

[GitHub](https://github.com/RahmaAshrafAli-Engineer)

---

<div align="center">

**Built with curiosity, analytical thinking, and a passion for AI.**

⭐ If you find this project interesting, feel free to explore the notebooks and share your feedback!

</div>
