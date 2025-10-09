# check_versions.py - Run this in VS Code terminal to verify installations
import fastapi
import uvicorn
import python_multipart
import pydantic
import numpy
import sklearn
import pandas
import joblib

print("=== VERSIONS IN VS CODE ENVIRONMENT ===")
print(f"FastAPI: {fastapi.__version__}")
print(f"Uvicorn: {uvicorn.__version__}")
print(f"Python-multipart: {python_multipart.__version__}")
print(f"Pydantic: {pydantic.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"Joblib: {joblib.__version__}")