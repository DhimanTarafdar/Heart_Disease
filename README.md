# Heart Disease Prediction using Machine Learning

This project aims to predict the likelihood of a patient having heart disease based on various medical attributes. The analysis is performed using a dataset from UCI Machine Learning Repository, and several classification models are trained and evaluated to find the best performing one.

## Project Overview

The main objective is to build a machine learning model that can accurately classify whether a patient has heart disease or not. This is a binary classification problem.

The complete analysis and model building process can be found in the Jupyter Notebook: `Heart_Disease.ipynb`.

## Dataset

The dataset used is `heart_disease_dataset.csv`. It contains 14 columns (attributes) including:
- `age`: Age of the patient
- `sex`: (1 = male; 0 = female)
- `cp`: Chest pain type
- `trestbps`: Resting blood pressure
- `chol`: Serum cholestoral in mg/dl
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting electrocardiographic results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: The slope of the peak exercise ST segment
- `ca`: Number of major vessels (0-3) colored by flourosopy
- `thal`: 3 = normal; 6 = fixed defect; 7 = reversable defect
- `target`: **(Target Variable)** 0 = no disease, 1 = disease

##Technologies & Libraries Used

- **Python**: Core programming language
- **Pandas**: For data manipulation and analysis
- **NumPy**: For numerical operations
- **Scikit-learn**: For machine learning models and evaluation
- **Matplotlib & Seaborn**: For data visualization
- **Jupyter Notebook**: For interactive coding and analysis

## ⚙️ How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DhimanTarafdar/Heart_Disease.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd Heart_Disease
    ```
3.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab
    ```
4.  **Launch Jupyter Notebook and open the file:**
    ```bash
    jupyter notebook Heart_Disease.ipynb
    ```

## 📈 Methodology

The project follows these key steps:
1.  **Data Loading & Initial Exploration:** Understanding the features and checking for missing values.
2.  **Exploratory Data Analysis (EDA):** Visualizing the data to find patterns and correlations between different features and the target variable.
3.  **Model Building:** Splitting the data into training and testing sets and training the following classification models:
    - `LogisticRegression`
    - `RandomForestClassifier`
    - `DecisionTreeClassifier`
4.  **Model Evaluation:** Comparing the models based on their **Accuracy** score to determine the most effective one.

## 📊 Results

The performance of the models on the test set was as follows:

| Model                    | Accuracy (%) |
| ------------------------ | ------------ |
| **Logistic Regression**  | **~87%**     |
| RandomForestClassifier   | ~85%         |
| DecisionTreeClassifier   | ~80%         |

Based on the evaluation, **Logistic Regression** was the best-performing model for this dataset.

## ✒️ Author

- **Dhiman Tarafdar** - [GitHub Profile](https://github.com/DhimanTarafdar)
