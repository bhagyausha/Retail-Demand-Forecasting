# Retail Demand Forecasting

## Project Description

This project focuses on forecasting retail demand using historical sales data. The goal is to build a predictive model that can forecast future sales for retail products, assisting businesses in inventory management, procurement, and supply chain decisions.

### Overview

Retail demand forecasting involves predicting the future sales of products based on historical sales data. Accurate forecasts help businesses optimize stock levels, minimize overstocking, and avoid stockouts. This project applies machine learning models to predict sales and optimize demand forecasting. The dataset includes various features such as historical sales data, pricing, stock levels, and date-related information.

## Features

- **Historical Sales Data**: Sales data from previous months to train the model.
- **Stock Level Information**: Information about the available stock.
- **Price Data**: Pricing of the products at different times.
- **Temporal Features**: Features like date, month, and day of the week to capture seasonal trends.

## Dataset

The dataset used in this project includes the following columns:

- **year**: Year of the sales record.
- **month**: Month of the sales record.
- **day**: Day of the sales record.
- **day_of_week**: Day of the week (e.g., Monday, Tuesday).
- **is_weekend**: Whether the date is a weekend or not (Boolean).
- **estoque**: Stock available for the product.
- **preco**: Price of the product.
- **rolling_sales**: Moving average of past sales to capture trends.
- **lag_sales**: Sales data from previous periods (lag features).
- **venda**: Actual sales volume.

The dataset also includes other temporal features and product information to improve model accuracy.

## Getting Started

To get started with this project, follow the steps below to set up the environment and install the necessary libraries.

### Prerequisites

Before you begin, ensure you have the following:

- Python 3.7 or higher
- Pandas, NumPy, Scikit-learn
- Matplotlib (for visualization)
- Jupyter Notebook or any IDE of your choice

### Installation

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/bhagyausha/Retail-Demand-Forecasting.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Retail-Demand-Forecasting
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Load Data**: Load the data from the provided CSV file or your own dataset.
2. **Preprocessing**: Preprocess the data, including filling missing values, scaling numerical features, and encoding categorical features.
3. **Model Training**: Split the data into training and test sets, and train a model (e.g., Linear Regression, Random Forest).
4. **Prediction**: Make predictions for future sales.
5. **Evaluation**: Evaluate the model's performance using metrics like RMSE or MAE.

Example code for making predictions:

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('path_to_your_data.csv')

# Preprocessing steps (e.g., filling missing values)
df['estoque'] = df['estoque'].fillna(0)

# Feature selection and training
X = df[['year', 'month', 'day', 'estoque', 'preco']]
y = df['venda']

# Train a model
model = LinearRegression()
model.fit(X, y)

# Predict future sales
predictions = model.predict(X)
df['predicted_sales'] = predictions

# Save the predictions
df.to_csv('processed_sales_data_with_predictions.csv', index=False)

```

## Model
This project uses a Linear Regression model for sales prediction, but other models such as Random Forest or XGBoost can be tested for better performance. 
The model is trained on historical data, and the features are carefully chosen to capture the temporal nature of sales trends.
  #### Model type: Linear Regression
  #### Features used: Date, stock, price, rolling, and lag sales.

## Contributing
Feel free to fork the repository, submit issues, and create pull requests if you'd like to contribute improvements or bug fixes!

## Project Structure
```
Retail-Demand-Forecasting/
│
├── data/                    # Directory for raw and processed data files
│   ├── raw/                 # Raw data files (original, unmodified data)
│   │   └── sales_data.csv  # Example raw sales data file
│   ├── processed/           # Processed data after cleaning and transformations
│   │   └── processed_sales_data.csv
│   └── predictions/         # Directory for files containing predictions
│       └── processed_sales_data_with_predictions.csv
│
├── notebooks/               # Jupyter notebooks for data exploration, analysis, and model development
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/                     # Source code (scripts for data processing, modeling, etc.)
│   ├── __init__.py
│   ├── data_preprocessing.py  # Script for cleaning and transforming raw data
│   ├── feature_engineering.py  # Script for creating new features like lag_sales, rolling_sales
│   ├── model.py              # Script for defining, training, and evaluating models
│   └── utils.py              # Utility functions (e.g., for evaluation metrics, saving models)
│
├── requirements.txt         # List of Python dependencies (e.g., pandas, scikit-learn, matplotlib)
├── README.md                # Project overview and instructions (this file)
├── LICENSE                  # License information (e.g., MIT License)
└── .gitignore               # Git ignore file (e.g., to ignore data, model, and environment files)
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
