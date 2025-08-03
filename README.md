# 📊 Predicting Customer Subscription using Bank Marketing Data

![Badge](https://img.shields.io/badge/Student-Nabonyimana%20Jospin-009688)
![Course](https://img.shields.io/badge/INSY8413-Intro_to_Big_Data-blue)
![Status](https://img.shields.io/badge/Project-Completed-success)

> A complete data analytics and machine learning project combining **Python**, **Google Colab**, and **Power BI**, submitted for the Capstone of *Introduction to Big Data Analytics (INSY 8413)*.

---

## 🎯 Project Summary

**Predicting customer subscription to term deposits using bank marketing data.**

This project leverages real-world structured data from a bank marketing campaign to build a predictive model that determines whether a client will subscribe to a term deposit based on their personal and interaction data. All required analytics tasks have been performed and visualized through interactive dashboards.

---

## 🧠 Objective

> According to the project assignment:

* 🩹 Perform **intensive data cleaning** in Python
* 📊 Generate **insightful visualizations** in Power BI
* 🤖 Apply a **machine learning model** to predict target outcome
* 📈 Deliver a **complete submission** via GitHub and report

---

## 🧰 Tools & Technologies Used

| Tool            | Role in Project                              |
| --------------- | -------------------------------------------- |
| 🐍 Python       | Data cleaning, preprocessing, model building |
| ☁️ Google Colab | Cloud-based Python execution                 |
| 📊 Power BI     | Interactive dashboards and visual insights   |
| 🧾 GitHub       | Project hosting, report, and documentation   |

---

## 📁 Contents of the Repository

| File/Folder                                  | Description                                   |
| -------------------------------------------- | --------------------------------------------- |
| `Bank_marketing_data.csv`                    | Raw dataset used for analysis                 |
| `final_cleaned_data.csv`                     | Cleaned dataset after preprocessing           |
| `Bank_marketing_cleaned_data_analysis.ipynb` | Google Colab notebook with full pipeline      |
| `PowerBI_Dashboard.pbix`                     | Fully interactive Power BI dashboard          |
| `screenshots/`                               | All relevant Power BI visual screenshots      |
| `README.md`                                  | This document: overview, methods, and results |

---

## 🚀 Project Stages Overview

### ✅ Part 1: Dataset & Project Setup

* Sector: **Banking / Marketing**
* Dataset: [Bank Marketing (UCI)](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
* Public, structured, and suitable for classification

### ✅ Part 2: Python Analysis

#### 🔧 Data Cleaning

**Purpose:** Ensure the dataset is accurate, consistent, and usable for analysis by handling missing values, formatting issues, and noise.

```python
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("Bank_marketing_data.csv")

# Replace 'unknown' with NaN
df.replace("unknown", np.nan, inplace=True)

# Convert to numeric
df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
df["day"] = pd.to_numeric(df["day"], errors="coerce")

# Drop rows with NaN or zero duration/balance
df_cleaned = df.dropna()
df_cleaned = df_cleaned[(df_cleaned["duration"] > 0) & (df_cleaned["balance"] > 0)]

# Reset index
df_cleaned.reset_index(drop=True, inplace=True)
```

**🗈 Screenshot of cleaned output:**

> 📸 `screenshots/data_cleaning_head.png`

---

#### 📊 Exploratory Data Analysis (EDA)

**Purpose:** Understand relationships between variables and identify useful patterns.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Subscription distribution
sns.countplot(data=df_cleaned, x='y')
plt.title("Target Variable: Subscribed or Not")
plt.show()
```

**🗈 Screenshot of EDA output:**

> 📸 `screenshots/eda_target.png`

Other charts:

* Job vs Subscription
* Education vs Subscription
* Duration Distribution
* Correlation Heatmap

---

### ✅ Part 3: Machine Learning Model

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Encode categorical variables
df_encoded = pd.get_dummies(df_cleaned, drop_first=True)
X = df_encoded.drop("y_yes", axis=1)
y = df_encoded["y_yes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

✅ Accuracy \~ **84%**

**🗈 Screenshot of model evaluation:**

> 📸 `screenshots/model_report.png`

---

### ✅ Part 4: Power BI Dashboard

> See visuals in `/screenshots/` folder and `.pbix` file

**Sample Visuals Created(From POWER BI):**

| No. | Visualization                     | Insight                                      |
| --- | --------------------------------- | -------------------------------------------- |
| 1️⃣ | Job vs Subscription               | Management and retired people subscribe more |
| 2️⃣ | Education vs Subscription         | Higher education tends to subscribe more     |
| 3️⃣ | Call Duration Histogram           | Longer calls often result in subscriptions   |
| 4️⃣ | Subscription Distribution Pie     | Majority of customers did not subscribe      |
| 5️⃣ | Month-wise Campaign Effectiveness | May and June show more campaign activity     |

**Slicers Included:** Month, Job, Marital Status, Subscription

---

## 🌍 Applications & Value

This project helps banks:

* 🎯 Target the right customers for term deposits
* 🧹 Understand customer characteristics
* 📈 Use data to support smarter marketing campaigns

---

## 👨‍🎓 Author

**Nabonyimana Jospin**
🎓 AUCA – Adventist University of Central Africa
📧 Email: <a href="mailto:jospinnabonyimana@gmail.com">[jospinnabonyimana@gmail.com](mailto:jospinnabonyimana@gmail.com)</a>

---

## 🏁 Academic Declaration & License

📚 This project was developed as a Capstone assignment for **INSY 8413 – Introduction to Big Data Analytics** at AUCA.

© 2025 Nabonyimana Jospin – All rights reserved.
