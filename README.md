# AI-Powered Retail Pricing Intelligence

## Project Overview

AI-Powered Retail Pricing Intelligence is an end-to-end Machine Learning project designed to predict retail product demand using historical sales data, product pricing, inventory levels, competitor pricing, discounts, promotions, weather conditions, and seasonal trends.

The system uses a **Random Forest Regressor** to predict product demand and supports data-driven pricing decisions through demand prediction and price optimization analysis.

The project integrates **MLflow** for experiment tracking, **FastAPI** for backend prediction services, **Streamlit** for the interactive user interface, and **Docker** for containerization.

The frontend and backend are containerized separately using Docker Compose, and their images have been published to Docker Hub.

## Project Objectives

- Analyze retail sales and pricing data.
- Perform data cleaning, preprocessing, and exploratory data analysis.
- Identify important features influencing product demand.
- Train and compare Machine Learning regression models.
- Evaluate and tune the selected prediction model.
- Predict retail product demand using a trained model.
- Analyze pricing strategies using predicted demand.
- Track Machine Learning experiments using MLflow.
- Build a FastAPI backend for demand prediction.
- Develop an interactive Streamlit dashboard.
- Containerize the frontend and backend using Docker.
- Publish the project source code on GitHub and Docker images on Docker Hub.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Final Model | Random Forest Regressor |
| Experiment Tracking | MLflow |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Development Tools | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |
| Container Registry | Docker Hub |

## Project Structure

```text
AI-Powered Retail Pricing Intelligence/
│
├── backend/
│   ├── DockerFile
│   ├── main.py
│   └── schemas.py
│
├── data/
│   ├── raw/
│   │   └── sales_data.csv
│   │
│   └── processed/
│       ├── cleaned_sales_data.csv
│       ├── feature_engineered_sales_data.csv
│       ├── model_results.csv
│       ├── model_tuning_results.csv
│       └── selected_features_data.csv
│
├── frontend/
│   ├── app.py
│   └── DockerFile
│
├── mlartifacts/
│
├── models/
│   ├── best_demand_model.pkl
│   └── demand_model.pkl
│
├── notebooks/
│   ├── Data Understanding.ipynb
│   ├── Data Cleaning.ipynb
│   ├── EDA.ipynb
│   ├── Feature Engineering.ipynb
│   ├── Feature Selection.ipynb
│   ├── Demand Prediction.ipynb
│   ├── Model Tuning.ipynb
│   ├── Model Saving.ipynb
│   └── Price Optimization.ipynb
│
├── .dockerignore
├── .gitignore
├── check_model.py
├── compose.yaml
├── mlflow_tracking.py
├── mlflow.db
├── requirements.txt
├── requirements-backend.txt
├── requirements-frontend.txt
└── README.md
```

**Note:** Virtual environments, notebook checkpoints, cached files, MLflow artifacts, and large trained model files may be excluded from the GitHub repository.

## Dataset Description

The project uses a retail sales dataset containing information about products, prices, inventory, promotions, and demand.

### Main Dataset Features

| Feature | Description |
|---|---|
| Date | Date of the retail record |
| Category | Product category |
| Region | Sales region |
| Inventory Level | Available stock quantity |
| Price | Product selling price |
| Discount | Discount offered |
| Weather Condition | Weather information |
| Promotion | Promotional activity |
| Competitor Pricing | Competitor's product price |
| Seasonality | Seasonal information |
| Epidemic | Epidemic-related indicator |
| Demand | Target variable for prediction |

**Target Variable:** `Demand`

The Machine Learning model estimates product demand based on the input features.

## Project Workflow

### 1. Data Understanding

**Notebook:** `Data Understanding.ipynb`

- Load and inspect the retail dataset.
- Identify rows and columns.
- Examine data types.
- Understand numerical and categorical features.
- Identify the target variable.

### 2. Data Cleaning

**Notebook:** `Data Cleaning.ipynb`

- Identify missing values.
- Check duplicate records.
- Handle inconsistent data.
- Prepare the cleaned dataset.
- Save the processed data as `cleaned_sales_data.csv`.

### 3. Exploratory Data Analysis (EDA)

**Notebook:** `EDA.ipynb`

- Analyze demand distribution.
- Examine price and demand relationships.
- Analyze product categories and regions.
- Study seasonal demand patterns.
- Generate visualizations for retail data analysis.

### 4. Feature Engineering

**Notebook:** `Feature Engineering.ipynb`

Additional features include:

- Year
- Month
- Day
- DayOfWeek
- Price_Difference
- Price_Difference_Percent
- Stock_Status
- Inventory_to_Order_Ratio

The resulting dataset is saved as `feature_engineered_sales_data.csv`.

### 5. Feature Selection

**Notebook:** `Feature Selection.ipynb`

- Identify relevant features for demand prediction.
- Analyze feature relationships.
- Prepare selected input features.
- Save the dataset as `selected_features_data.csv`.

### 6. Demand Prediction

**Notebook:** `Demand Prediction.ipynb`

The following Machine Learning algorithms were explored:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

The models are trained to predict retail product demand.

Model comparison results are stored in `model_results.csv`.

### 7. Model Tuning

**Notebook:** `Model Tuning.ipynb`

- Tune model hyperparameters.
- Compare model configurations.
- Evaluate model performance.
- Select the best-performing configuration.

Model tuning results are stored in `model_tuning_results.csv`.

### 8. Model Evaluation

The models are evaluated using:

- **MAE (Mean Absolute Error):** Measures average absolute prediction error.
- **RMSE (Root Mean Squared Error):** Measures prediction error with greater weight on large errors.
- **R² Score:** Measures how well the model explains variation in demand.

**Selected Model:** Random Forest Regressor

### 9. Model Saving

**Notebook:** `Model Saving.ipynb`

The trained Machine Learning pipelines are saved for reuse.

Saved model files include:

- `best_demand_model.pkl`
- `demand_model.pkl`

These models can be loaded using Joblib for demand prediction.

### 10. Price Optimization

**Notebook:** `Price Optimization.ipynb`

Price optimization analysis uses predicted demand to compare different product pricing scenarios.

**Estimated Revenue = Selling Price × Predicted Demand**

This helps evaluate potential pricing strategies.

## MLflow Experiment Tracking

MLflow is used to track Machine Learning experiments.

Features include:

- Logging model parameters.
- Tracking evaluation metrics.
- Comparing experiment runs.
- Storing model artifacts.
- Managing model experiments.
- Supporting model registration.

The project includes:

- `mlflow_tracking.py`
- `mlflow.db`
- `mlartifacts/`

To open the MLflow interface using the local SQLite tracking database:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

**MLflow URL:** http://127.0.0.1:5000

## FastAPI Backend

FastAPI is used to provide prediction services.

The backend:

- Loads the saved Machine Learning model.
- Receives product and pricing information.
- Validates input data using Pydantic schemas.
- Generates demand predictions.
- Returns prediction results in JSON format.

**Backend Files:**

- `backend/main.py`
- `backend/schemas.py`

**Prediction Endpoint:**

`POST /predict-demand`

**API Documentation:**

http://localhost:8000/docs

### Run FastAPI

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## Streamlit Frontend

Streamlit provides an interactive interface for retail demand prediction.

**Frontend File:** `frontend/app.py`

### Features

- Product category selection.
- Region selection.
- Inventory level input.
- Price and discount input.
- Competitor pricing input.
- Weather and seasonality selection.
- Promotion information.
- FastAPI integration.
- Predicted demand display.

### Run Streamlit

Open another terminal and run:

```bash
streamlit run frontend/app.py
```

**Streamlit URL:** http://localhost:8501

The FastAPI backend must be running and accessible to the frontend.

## System Architecture

```text
Retail Sales Dataset
         |
         v
Data Understanding
         |
         v
Data Cleaning
         |
         v
Exploratory Data Analysis
         |
         v
Feature Engineering
         |
         v
Feature Selection
         |
         v
Model Training and Evaluation
         |
         v
Model Tuning
         |
         +------> MLflow Experiment Tracking
         |
         v
Random Forest Regressor
         |
         v
Saved Machine Learning Model
         |
         v
FastAPI Backend
         ^
         |
         v
Streamlit Frontend
         |
         v
Predicted Product Demand
```

## Installation and Setup

### Prerequisites

- Python 3.11
- Git
- VS Code
- Docker Desktop
- Docker Compose

### Step 1: Clone the Repository

```bash
git clone https://github.com/Vishnu198L/AI-Powered-Retail-Pricing-Intelligence.git
cd AI-Powered-Retail-Pricing-Intelligence
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The repository also contains separate dependency files:

- `requirements-backend.txt`
- `requirements-frontend.txt`

### Step 4: Prepare the Trained Model

Ensure that the trained model file is available at the location configured in `backend/main.py`.

The backend requires the saved model to generate predictions.

If large model files are excluded from GitHub, they must be supplied separately or regenerated using the training workflow.

### Step 5: Start FastAPI

```bash
uvicorn backend.main:app --reload
```

### Step 6: Start Streamlit

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501 to use the application.

## Docker Containerization

The application is containerized using Docker and Docker Compose.

Two Docker images are used:

**Frontend Container**

- Application: Streamlit
- Port: `8501`
- Purpose: Collect product inputs and display predictions.

**Backend Container**

- Application: FastAPI
- Port: `8000`
- Purpose: Load the trained model and generate predictions.

Docker Compose manages both containers and provides communication between them.

### Build and Run Containers

Make sure Docker Desktop is running.

From the project root directory:

```bash
docker compose up --build
```

### Check Running Containers

```bash
docker compose ps
```

### Access the Application

| Service | URL |
|---|---|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| FastAPI Documentation | http://localhost:8000/docs |

### Stop Containers

```bash
docker compose down
```

## Docker Hub Repositories

The frontend and backend images have been published to Docker Hub.

### Frontend Image

**Repository:** `vishnu178l/retail-pricing-frontend`

https://hub.docker.com/r/vishnu178l/retail-pricing-frontend

### Backend Image

**Repository:** `vishnu178l/retail-pricing-backend`

https://hub.docker.com/r/vishnu178l/retail-pricing-backend

## Sample Prediction Input

| Feature | Example Value |
|---|---|
| Category | Electronics |
| Region | South |
| Inventory Level | 100 |
| Price | 500 |
| Discount | 10 |
| Weather Condition | Sunny |
| Promotion | Yes |
| Competitor Pricing | 480 |
| Seasonality | Summer |
| Epidemic | No |
| Year | 2026 |
| Month | 10 |
| Day | 14 |
| DayOfWeek | 2 |
| Price_Difference | 20 |
| Price_Difference_Percent | 4.1667 |
| Stock_Status | Normal |
| Inventory_to_Order_Ratio | 1.5 |

The model processes these features and returns an estimated demand value.

The prediction represents estimated demand based on patterns learned from the training dataset. It does not guarantee actual sales.

## Project Features

- Retail demand prediction.
- Data cleaning and preprocessing.
- Exploratory Data Analysis.
- Feature engineering and selection.
- Machine Learning model comparison.
- Random Forest Regression.
- Model tuning and evaluation.
- Model saving using Joblib.
- MLflow experiment tracking.
- Price optimization analysis.
- FastAPI backend integration.
- Interactive Streamlit frontend.
- Docker containerization.
- Docker Compose integration.
- Docker Hub image publishing.
- GitHub source code management.

## Limitations

1. Model performance depends on the quality of historical data.
2. Sudden market changes may reduce prediction accuracy.
3. The system depends on the available input features.
4. Predictions may be less reliable for unseen product categories.
5. Pricing analysis depends on the accuracy of predicted demand.

## Future Enhancements

1. Integrate real-time retail sales and competitor pricing data.
2. Improve model performance using additional datasets.
3. Implement automated model retraining.
4. Deploy the application to a cloud platform.
5. Extend price optimization using profit margins and business constraints.

## Conclusion

The **AI-Powered Retail Pricing Intelligence** project demonstrates an end-to-end Machine Learning workflow for retail demand prediction.

The project includes data preprocessing, exploratory data analysis, feature engineering, Random Forest Regression, model evaluation, MLflow experiment tracking, FastAPI backend development, Streamlit frontend integration, and Docker containerization.

The system provides a practical foundation for retail demand prediction and data-driven pricing analysis.

## GitHub Repository

https://github.com/Vishnu198L/AI-Powered-Retail-Pricing-Intelligence