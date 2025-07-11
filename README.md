# salary-prediction-app
A Streamlit-based machine learning app to predict salaries using ensemble models like Random Forest, Gradient Boosting, and AdaBoost.
# 💼 Salary Prediction Web App

This is a machine learning project to predict salaries based on position levels using **ensemble learning models**.  
The app is built with **Streamlit** and includes an interactive GUI for users to explore the data and predict salaries.

---

## 🚀 Features

- 📂 Upload your own CSV file or use the default one
- 🧠 Choose from 3 ML models:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - AdaBoost Regressor
- 📈 Visualize the predicted curve vs actual data
- 🎛️ Easy slider input for level
- 🔢 Instant prediction with interactive output

---

## 📁 Dataset Used

**File:** `Position_Salaries.csv`  
**Columns:**
- `Position`: Job title
- `Level`: Numerical rank (1 to 10)
- `Salary`: Actual salary in ₹

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

---

## 📦 Installation & Run (Locally)

```bash
# Clone the repo
git clone https://github.com/yourusername/salary-prediction-app.git

# Move into the project folder
cd salary-prediction-app

# Install required libraries
pip install -r requirements.txt

# Run the app
streamlit run app.py
