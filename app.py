import streamlit as st
import google.generativeai as genai
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


import io

# Configure API key directly
genai.configure(api_key="AIzaSyDoR10wPWSnCCLXHZWWrlrAg7XCXFzzpx8")  # Replace with your actual API key

# Function to generate dataset based on a text prompt using Google Generative AI
def generate_dataset_from_text(text):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(["Generate a dataset in CSV format based on the following text without explanation, just data and I want exact 200 rows and 5 columns .if you dont have data dont leave it blank put there as NaN also strictly don't give ```csv ``` or '''  ''' with dataset :", text])
    return response.text

# Function to write requirements to requirements.txt
def write_requirements_file():
    requirements = """
pandas
scikit-learn
streamlit
google-generativeai
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())

# Function to preprocess dataset
def preprocess_dataset(csv_data, task_type):
    try:
        # Read the CSV data from a string instead of a file
        df = pd.read_csv(io.StringIO(csv_data))  # Use StringIO to read from string data
    except pd.errors.EmptyDataError:
        st.error("No data to read from the CSV.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error reading CSV data: {e}")
        return None, None, None, None, None

    # If the DataFrame is empty, return early
    if df.empty:
        st.error("The DataFrame is empty after reading the CSV.")
        return None, None, None, None, None

    # Replace "None" strings with actual NaN values
    df.replace("None", pd.NA, inplace=True)

    # Separate features and target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Identify categorical and numeric columns
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Create a preprocessing pipeline for numeric features (imputation + scaling)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Create a preprocessing pipeline for categorical features (imputation + encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine numeric and categorical transformations into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Apply the transformations to X
    X_transformed = preprocessor.fit_transform(X)

    # If it's a classification task, label encode the target variable
    if task_type == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor, X.columns.tolist()

# Function to train models and select the best one with hyperparameter tuning
def train_models(X_train, y_train, X_test, y_test, task_type):
    models = {
        "Decision Tree": DecisionTreeClassifier() if task_type == 'classification' else DecisionTreeRegressor(),
        "Support Vector Machine": SVC() if task_type == 'classification' else SVR(),
        "K-Nearest Neighbors": KNeighborsClassifier() if task_type == 'classification' else KNeighborsRegressor(),
        "Logistic Regression": LogisticRegression(max_iter=1000) if task_type == 'classification' else None,
    }

    best_model = None
    best_model_name = ""
    best_score = -float('inf')

    for model_name, model in models.items():
        if model:
            param_grid = {}
            if model_name == "Decision Tree":
                param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
            elif model_name == "Support Vector Machine":
                param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            elif model_name == "K-Nearest Neighbors":
                param_grid = {'n_neighbors': [3, 5, 7]}
            elif model_name == "Logistic Regression":
                param_grid = {'C': [0.1, 1, 10]}
            grid_search = GridSearchCV(model, param_grid, scoring='accuracy' if task_type == 'classification' else 'r2', cv=5)
            grid_search.fit(X_train, y_train)

            # Predict on the test set
            y_pred = grid_search.predict(X_test)

            # Calculate score
            score = accuracy_score(y_test, y_pred) if task_type == 'classification' else r2_score(y_test, y_pred)

            # Track the best model based on score
            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
                best_model_name = model_name

    return best_model, best_model_name, best_score

# Function to save the best model
def save_best_model(model):
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Function to generate loading code
def generate_loading_code(filename, feature_names):
    # Create input statements based on the feature names
    input_statements = "\n".join([f"{name} = float(input('Enter value for {name}: '))" for name in feature_names])
    
    # Determine the prediction statement
    prediction_statement = "prediction = model.predict([[{}]])".format(", ".join(feature_names))
    
    # Generate the loading code
    code = f"""
import pickle

def load_model():
    with open('{filename}', 'rb') as f:
        model = pickle.load(f)
    return model

def predict():
    model = load_model()
    
    {input_statements}  # Get user inputs
    {prediction_statement}  # Make prediction
    
    print("Predicted output:", prediction)

if __name__ == "__main__":
    predict()
"""
    
    with open("load_model.py", "w") as f:
        f.write(code)

# Streamlit App
st.set_page_config(page_title="AI-Generated ML System", layout="wide")
st.title("🎉 AI-Generated Machine Learning System")
st.markdown(""" 
Welcome to the AI-Generated Machine Learning System! Here, you can generate datasets, train machine learning models, and save the best model for future use. Please follow the instructions below to get started. 
""")

# Sidebar for inputs
st.sidebar.header("Project Configuration")
text_prompt = st.sidebar.text_area("Describe your project requirements:", height=150)
task_type = st.sidebar.selectbox("Select Task Type", options=["regression", "classification"])

# Button to generate project
if st.sidebar.button("Generate Project"):
    if text_prompt:
        try:
            # Generate dataset from the text prompt
            csv_data = generate_dataset_from_text(text_prompt)

            # Check if the generated CSV data is valid before proceeding
            if not csv_data.strip():  # Check if csv_data is empty or only whitespace
                st.error("Generated CSV data is empty.")
            else:
                
                # Convert raw data to DataFrame for display and further processing
                try:
                    st.subheader("Generated Dataset:")
                    df_display = pd.read_csv(io.StringIO(csv_data))  # Read CSV from string
                    st.dataframe(df_display)
                    
                    # Preprocess the dataset
                    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_dataset(csv_data, task_type)

                    # Proceed only if preprocessing was successful
                    if X_train is not None:
                        # Train models and select the best one
                        best_model, best_model_name, best_score = train_models(X_train, y_train, X_test, y_test, task_type)

                        # Save the model
                        save_best_model(best_model)

                        # Generate loading code
                        generate_loading_code("best_model.pkl", feature_names)

                        # Write requirements file
                        write_requirements_file()

                        # Display success message
                        st.success(f"Successfully trained {best_model_name} with a score of {best_score:.2f}. Model saved as 'best_model.pkl'. Loading code saved as 'load_model.py'. Requirements file created as 'requirements.txt'.")
                        with open("best_model.pkl", "rb") as f:
                         st.download_button(label="Download Best Model (best_model.pkl)", data=f, file_name="best_model.pkl")

                        with open("load_model.py", "rb") as f:
                         st.download_button(label="Download Loading Code (load_model.py)", data=f, file_name="load_model.py")

                        with open("requirements.txt", "rb") as f:
                         st.download_button(label="Download Requirements (requirements.txt)", data=f, file_name="requirements.txt")

                except Exception as e:
                    st.error(f"Error processing the dataset: {e}")
        except Exception as e:
            st.error(f"Error generating dataset: {e}")
    else:
        st.warning("Please enter your project requirements in the text area.")
