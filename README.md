[README_booking.md](https://github.com/user-attachments/files/33962209/README_booking.md)
# Hotel Booking Machine Learning Project

This project uses machine learning to analyze hotel booking data, predict booking cancellation status, and estimate the average booking price. It includes data analysis, preprocessing, model training, evaluation, and a Streamlit web app.

## Features

- Exploratory Data Analysis (EDA) with charts and correlation analysis
- Date preprocessing: extracting year, month, and day from reservation dates
- Ordinal encoding for selected categorical columns
- Min-Max normalization and Standardization
- Missing-value imputation using the median
- Classification models to predict whether a booking will be canceled
- Regression models to estimate average price
- Cross-validation and hyperparameter tuning
- Evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC, MAE, RMSE, and R²
- Streamlit interface for making predictions

## Dataset

The notebook loads a CSV file named:

```text
booking.csv
```

Place this file in the same folder as `ML booking.ipynb` before running the notebook.

The target for classification is `booking status`. The regression task predicts `average price`.

## Machine Learning Models

The notebook compares classification algorithms including:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

For regression, it compares multiple regression models, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor, where defined in the notebook.

A Random Forest classifier is tuned using `GridSearchCV` with 5-fold cross-validation.

## Libraries Used

- **NumPy** — numerical operations
- **Pandas** — loading and manipulating tabular data
- **Matplotlib** — plotting and charts
- **Seaborn** — statistical visualizations
- **Scikit-learn** — preprocessing, ML models, cross-validation, tuning, and evaluation
- **Joblib** — saving and loading trained models
- **Streamlit** — interactive prediction app

The notebook also uses Python standard-library modules such as `os`, `json`, `warnings`, and `base64`; these do not need separate installation.

## Installation

Install the required packages:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn joblib streamlit notebook
```

## How to Run

1. Clone or download this repository.
2. Put `booking.csv` in the project folder.
3. Install the packages listed above.
4. Open the notebook:

   ```bash
   jupyter notebook
   ```

5. Run `ML booking.ipynb` cells in order to explore the data, train models, and save the model artifacts.
6. If `streamlit_app.py` and the required files in `MODEL_FILES/` have been created, launch the app:

   ```bash
   streamlit run streamlit_app.py
   ```

## Model Files

The notebook saves model and schema files in the `MODEL_FILES/` directory for use by the Streamlit app. Run the relevant notebook cells before launching the app if those files are not already available.

## Note

This project is for educational purposes. Model performance depends on the dataset and the way the notebook is executed. Check that the saved preprocessing objects and model inputs match before using the app for predictions.
