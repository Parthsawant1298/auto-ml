import os
import streamlit as st
import google.generativeai as genai
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import io
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
import subprocess

# Configure API key securely
genai.configure(api_key="AIzaSyDoR10wPWSnCCLXHZWWrlrAg7XCXFzzpx8")
kaggle_api = KaggleApi()
kaggle_api.authenticate()  # Authenticate with Kaggle API using your credentials


def download_kaggle_dataset(query):
    dataset_folder = "datasets"
    
    try:
        # Clear the dataset folder before downloading the new dataset
        if os.path.exists(dataset_folder):
            for file_name in os.listdir(dataset_folder):
                file_path = os.path.join(dataset_folder, file_name)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove directory
                except Exception as e:
                    st.error(f"Error deleting file {file_name}: {e}")

        # Search for datasets matching the query
        datasets = kaggle_api.dataset_list(search=query)
        if datasets:
            dataset = datasets[0]  # Get the first result
            dataset_name = dataset.ref  # Dataset reference
            
            # Download the dataset files to the dataset folder
            kaggle_api.dataset_download_files(dataset_name, path=dataset_folder, unzip=True)
            
            # List the files in the dataset folder to find the actual downloaded file
            downloaded_files = os.listdir(dataset_folder)
            
            # Return the first CSV file found
            for file in downloaded_files:
                if file.endswith('.csv'):
                    return file  # Just return the file name without the path
        else:
            return None
    except Exception as e:
        st.error(f"Error searching for Kaggle datasets: {e}")
        return None

# Function to generate dataset based on a text prompt using Google Generative AI
def generate_dataset_from_text(text):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([ 
        f"Generate a dataset in CSV format based on the following text without explanation, "
        f"just data and I want 200 rows and 5 columns, avoid repeating data both numeric as well as "
        f"categorical also strictly don't give ```csv ``` or '''  ''' with dataset : {text}."
    ])
    return response.text

# Function to write requirements to requirements.txt
def write_requirements_file():
    requirements = """
pickle
streamlit
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())

def create_venv():
    venv_folder = "venv"
    if not os.path.exists(venv_folder):
        st.text("Creating a virtual environment...")
        subprocess.run(["python", "-m", "venv", venv_folder])
        
        # Install required packages in the virtual environment
        subprocess.run([os.path.join(venv_folder, 'Scripts', 'pip'), 'install', '-r', 'requirements.txt'])

    return venv_folder
# Function to preprocess dataset
def preprocess_dataset(df, task_type):
    if df.empty:
        st.error("The DataFrame is empty after reading the CSV.")
        return None, None, None, None, None

    df.replace("None", pd.NA, inplace=True)

    # Separate features and target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    X_transformed = preprocessor.fit_transform(X)

    if task_type == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(y)

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

            y_pred = grid_search.predict(X_test)
            score = accuracy_score(y_test, y_pred) if task_type == 'classification' else r2_score(y_test, y_pred)

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
    input_statements = "\n".join([f"{name} = float(input('Enter value for {name}: '))" for name in feature_names])
    prediction_statement = f"prediction = model.predict([[{', '.join(['str(' + name + ')' for name in feature_names])}]])"
    code = f"""
import pickle

def load_model():
    with open('{filename}', 'rb') as f:
        model = pickle.load(f)
    return model

def predict():
    model = load_model()
    
    {input_statements}
    {prediction_statement}
    
    print("Predicted output:", prediction)

if __name__ == "__main__":
    predict()
"""
    
    with open("load_model.py", "w") as f:
        f.write(code)

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
        # First, try to download dataset from Kaggle
        downloaded_file_name = download_kaggle_dataset(text_prompt)
        
        if downloaded_file_name:
            st.success("Dataset found on Kaggle and downloaded.")
            # Read the downloaded dataset directly into a variable
            dataset_path = os.path.join("C:\\Users\\parth sawant\\Desktop\\ml major\\datasets", downloaded_file_name)
            df = pd.read_csv(dataset_path)
            st.subheader("Kaggle Dataset:")
            st.dataframe(df)  # Display the DataFrame in a nice format

            try:
                # Proceed with the Kaggle dataset
                st.subheader("Preprocessing Data...")
                csv_data = df.to_csv(index=False)
                X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_dataset(df, task_type)

                if X_train is not None and X_test is not None:
                    st.subheader("Training Models...")
                    best_model, best_model_name, best_score = train_models(X_train, y_train, X_test, y_test, task_type)
                    st.success(f"Best Model: {best_model_name} with Score: {best_score:.2f}")

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
                st.error(f"Error processing the Kaggle dataset: {e}")
        else:
            # If no dataset was found, generate a new dataset
            st.warning("No Kaggle dataset found, generating a new dataset...")
            generated_data = generate_dataset_from_text(text_prompt)
            generated_df = pd.read_csv(io.StringIO(generated_data))

            st.subheader("Generated Dataset:")
            st.dataframe(generated_df)  # Display the generated DataFrame

            try:
                # Preprocess the generated dataset
                X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_dataset(generated_df, task_type)

                if X_train is not None and X_test is not None:
                    st.subheader("Training Models...")
                    best_model, best_model_name, best_score = train_models(X_train, y_train, X_test, y_test, task_type)
                    st.success(f"Best Model: {best_model_name} with Score: {best_score:.2f}")

                    save_best_model(best_model)
                    st.success("Model saved successfully!")

                    # Generate loading code
                    generate_loading_code("best_model.pkl", feature_names)
                    st.success("Loading code generated successfully!")

                    with open("best_model.pkl", "rb") as file:
                        st.download_button("Download Loading Code", file, "best_model.pkl")
                    # Provide download link for loading code
                    with open("load_model.py", "rb") as file:
                        st.download_button("Download Loading Code", file, "load_model.py")
                    with open("requirements.txt", "rb") as file:
                        st.download_button("Download Loading Code", file, "requirements.txt")
            except Exception as e:
                st.error(f"Error processing the generated dataset: {e}")
