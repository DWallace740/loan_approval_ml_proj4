# Model Iteration Documentation

## Initial Model: Logistic Regression

### Implementation:
- Used `LogisticRegression` from `sklearn`
- Data was preprocessed using `StandardScaler`
- Training and testing split: 75% train, 25% test

### Performance:
- **Training Accuracy**: ~78%
- **Testing Accuracy**: ~78%
- **Findings**:
  - The model reached a decent accuracy but did not converge due to reaching the iteration limit.
  - A `ConvergenceWarning` was issued, indicating the solver failed to converge.
  - The model may benefit from increasing `max_iter` or adjusting feature scaling.
  - Potential risk of underfitting due to stopping early.

---

## Second Model: Random Forest Classifier

### Implementation:
- Used `RandomForestClassifier` with 100 estimators
- Feature importance visualization added
- Preprocessed data using `StandardScaler`

### Performance:
- **Feature Importance**: 
  - Most important features: `Credit Score`, `Loan Amount Requested`, `Annual Income`
  - Less important: `Education`, `Employment Status`
- **Confusion Matrix**:
  - Improved recall for high-risk loans
- **Findings**:
  - More robust classification with improved generalization.

---

## Third Model: Neural Network (Deep Learning)

### Implementation:
- Three hidden layers:
  - **Layer 1**: 128 neurons, ReLU activation
  - **Layer 2**: 64 neurons, ReLU activation
  - **Layer 3**: 32 neurons, Tanh activation
- Output layer with sigmoid activation

### Performance:
- **Accuracy**: **76.3%**
- **Loss**: **0.8751**
- **Training Trend**:
  - Loss steadily decreased
  - Accuracy improved over epochs
- **Findings**:
  - Lower accuracy compared to other models.
  - May need further hyperparameter tuning.

---

## Summary & Next Steps

- **Logistic Regression**: Strong performance but potential overfitting.
- **Random Forest**: Balanced trade-off with interpretability.
- **Neural Network**: Needs further optimization (layer tuning, epochs, dropout, etc.).
