
import joblib
from pathlib import Path

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent

# Updated saved model path
model_path = BASE_DIR / "models" / "demand_model.pkl"

print("Checking saved demand prediction model...")
print("Model path:", model_path)

if not model_path.exists():
    print("\nERROR: Model file not found!")
    print("Please check that models/demandmodel.pkl exists.")

else:
    try:
        # Load the saved model
        model = joblib.load(model_path)

        print("\nModel loaded successfully!")
        print("Model type:", type(model))

        print("\nModel pipeline:")

        if hasattr(model, "steps"):
            for step_name, step_object in model.steps:
                print(f"\nStep: {step_name}")
                print("Type:", type(step_object))
                print(step_object)
        else:
            print(model)

        print("\nExpected feature names:")
        print(
            getattr(
                model,
                "feature_names_in_",
                "Not available"
            )
        )

        print("\nExpected number of features:")
        print(
            getattr(
                model,
                "n_features_in_",
                "Not available"
            )
        )

        print("\nModel verification completed!")

    except Exception as e:
        print("\nERROR: Failed to load the model.")
        print("Details:", e)
