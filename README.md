# Loan Approval Prediction Using Machine Learning

## Overview
This project aims to predict loan approval status based on applicant financial data using various machine learning models, including logistic regression, random forest, and deep learning. The dataset consists of multiple financial and demographic factors, and we leverage data preprocessing, feature engineering, and model optimization to improve performance.

## Authors
- **Daena Wallace**
- **Avery Javier**

## Technologies Used
- Python
- Pandas for data manipulation
- Scikit-Learn for machine learning modeling
- TensorFlow/Keras for deep learning
- SQLite for database management
- Matplotlib & Seaborn for data visualization
- Jupyter Notebook for development

## Repository Structure
loan-approval-ml/ │── .gitignore # Git ignore file to exclude unnecessary files
│── Data_cleanup.ipynb # Data preprocessing and cleaning notebook
│── LoanApproval_Model.ipynb # Machine learning model implementation
│── LoanApproval_ML.h5 # Saved deep learning model
│── Model_Iteration.md # Documentation of model improvements
│── README.md # Project documentation
│── loan_dataset.csv # Original dataset
│── loan_dataset.db # SQLite database storing dataset
│── requirements.txt # Dependencies list

## Data Processing
- The dataset was loaded and stored in an SQLite database for structured data retrieval.
- Categorical variables were converted to numerical using one-hot encoding (`pd.get_dummies`).
- Feature scaling was performed using `StandardScaler` for better model convergence.

## Models Implemented
### **1. Logistic Regression**
- Baseline model with `LogisticRegression` from `sklearn`.
- Standard scaling was applied before training.
- Performance: **Training Accuracy: ~78%, Testing Accuracy: ~78%**.
- Identified underfitting due to convergence warning.

### **2. Random Forest Classifier**
- Implemented `RandomForestClassifier` to capture non-linear relationships.
- Evaluated feature importance and identified key predictors.
- Improved model accuracy compared to logistic regression.

### **3. Deep Learning Model (Neural Network)**
- Designed a sequential neural network using TensorFlow/Keras.
- Architecture: **3 hidden layers (128, 64, 32 units) with ReLU and Tanh activation**.
- Achieved **training accuracy ~92% and testing accuracy ~76%**.
- Demonstrated improved learning but still faced overfitting challenges.

## Results
- **Feature Importance Analysis**: Key predictors include **Credit Score, Loan Amount Requested, and Annual Income**.
- **Performance Metrics**:
  - Logistic Regression: **78% accuracy**
  - Random Forest: **Improved accuracy over logistic regression**
  - Neural Network: **92% training accuracy but 76% test accuracy (possible overfitting)**

## Next Steps
- **Hyperparameter tuning**: Adjust parameters for random forest and neural network.
- **Feature Engineering**: Create new features to enhance model performance.
- **Address Overfitting**: Implement dropout and regularization in the neural network.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/DWallace740/loan_approval_ml_proj4.git
2. Navigate to the project directory:
    ```bash
    cd loan-approval-ml
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
4. Run the Jupyter Notebook:
    ```bash
jupyter notebook LoanApproval_Model.ipynb

## Resources and Support
- Scikit-Learn Documentation: https://scikit-learn.org/
- TensorFlow/Keras: https://www.tensorflow.org/
- SQLite Documentation: https://www.sqlite.org/docs.html

## License
This project is for educational purposes only.
