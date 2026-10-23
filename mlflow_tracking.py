import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. MLflow Tracking Server and Experiment
# ============================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

experiment_name = "AI-Powered Retail Pricing Intelligence v2"

mlflow.set_experiment(experiment_name)

print("MLflow experiment selected successfully.")


# ============================================================
# 2. Load Feature-Engineered Dataset
# ============================================================

df = pd.read_csv(
    "data/processed/feature_engineered_sales_data.csv"
)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 3. Prepare Date Column
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    "Date"
).reset_index(drop=True)

print("Data sorted chronologically.")


# ============================================================
# 4. Define Features and Target
# ============================================================

feature_columns = [
    "Category",
    "Region",
    "Inventory Level",
    "Price",
    "Discount",
    "Weather Condition",
    "Promotion",
    "Competitor Pricing",
    "Seasonality",
    "Epidemic",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "Price_Difference",
    "Price_Difference_Percent",
    "Stock_Status",
    "Inventory_to_Order_Ratio"
]

X = df[feature_columns].copy()

y = df["Demand"].copy()

print("Features and target prepared.")
print("Number of features:", len(feature_columns))


# ============================================================
# 5. Chronological Train-Test Split
# ============================================================

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()


print("\nTrain-Test Split")
print("----------------------------")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 6. Identify Categorical and Numerical Features
# ============================================================

categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=[np.number]
).columns.tolist()


print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)


# ============================================================
# 7. Create Preprocessor
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


# ============================================================
# 8. Define Models
# ============================================================

models = {

    "Linear Regression": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    ),

    "Gradient Boosting": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            )
        ]
    )
}


# ============================================================
# 9. Train and Track All Models
# ============================================================

results = []


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    with mlflow.start_run(
        run_name=model_name
    ):

        # ----------------------------------------------------
        # Train Model
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        print("Model training completed.")


        # ----------------------------------------------------
        # Generate Predictions
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        print("Predictions generated.")


        # ----------------------------------------------------
        # Calculate Metrics
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        r2 = r2_score(
            y_test,
            y_pred
        )


        # ----------------------------------------------------
        # Log Common Parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model_type",
            model_name
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "number_of_features",
            len(feature_columns)
        )

        mlflow.log_param(
            "train_samples",
            len(X_train)
        )

        mlflow.log_param(
            "test_samples",
            len(X_test)
        )


        # ----------------------------------------------------
        # Log Model-Specific Parameters
        # ----------------------------------------------------

        if model_name == "Random Forest":

            rf_model = model.named_steps["model"]

            mlflow.log_param(
                "n_estimators",
                rf_model.n_estimators
            )

            mlflow.log_param(
                "random_state",
                rf_model.random_state
            )

            mlflow.log_param(
                "n_jobs",
                rf_model.n_jobs
            )


        elif model_name == "Gradient Boosting":

            gb_model = model.named_steps["model"]

            mlflow.log_param(
                "n_estimators",
                gb_model.n_estimators
            )

            mlflow.log_param(
                "learning_rate",
                gb_model.learning_rate
            )

            mlflow.log_param(
                "max_depth",
                gb_model.max_depth
            )

            mlflow.log_param(
                "random_state",
                gb_model.random_state
            )


        # ----------------------------------------------------
        # Log Metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "RMSE",
            rmse
        )

        mlflow.log_metric(
            "R2",
            r2
        )


        # ----------------------------------------------------
        # Log Model
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="demand_model",
            serialization_format="cloudpickle"
        )


        # ----------------------------------------------------
        # Store Results
        # ----------------------------------------------------

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })


        # ----------------------------------------------------
        # Print Results
        # ----------------------------------------------------

        print("\nModel Performance")
        print("----------------------------")
        print("MAE :", mae)
        print("RMSE:", rmse)
        print("R2  :", r2)

        print(
            "\nMLflow tracking completed for:",
            model_name
        )


# ============================================================
# 10. Model Comparison
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
).reset_index(drop=True)


print("\n\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 11. Best Model
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Model:",
    best_model["Model"]
)

print(
    "MAE:",
    best_model["MAE"]
)

print(
    "RMSE:",
    best_model["RMSE"]
)

print(
    "R2:",
    best_model["R2"]
)


# ============================================================
# 12. Completion Message
# ============================================================

print("\n" + "=" * 60)
print("ALL MODELS TRACKED SUCCESSFULLY")
print("=" * 60)

print(
    "\nOpen MLflow UI at:"
)

print(
    "http://127.0.0.1:5000"
)