# ReneWind — Wind Turbine Predictive Maintenance with Neural Networks

## Project Overview

ReneWind is a predictive-maintenance machine learning project focused on identifying potential wind-turbine generator failures before equipment breaks down.

The project uses sensor-derived features and neural networks to classify each observation as:

- `0` — No Failure
- `1` — Failure

The business objective is to detect failures early so maintenance teams can repair equipment before a complete breakdown, reducing expensive generator replacement costs and unplanned downtime.

This project covers:

- Exploratory data analysis
- Missing-value treatment
- Class-imbalance analysis
- Train/validation/test preprocessing without data leakage
- Neural-network model building with SGD and Adam
- Dropout and class-weight experiments
- Performance comparison across multiple neural-network configurations
- Final-model selection
- Test-set evaluation
- Business recommendations for predictive maintenance

---

## Business Context

ReneWind wants to improve wind-energy production by predicting generator failures using sensor data.

The cost structure makes model errors especially important:

- **True Positive:** Failure correctly detected → repair cost
- **False Negative:** Failure not detected → generator replacement cost
- **False Positive:** Failure incorrectly predicted → inspection cost

The project assumes:

```text
Replacement Cost > Repair Cost > Inspection Cost
```

Because missed failures are the most expensive outcome, recall for the failure class is important. At the same time, too many false alarms create unnecessary inspection costs, so the project uses **F1-score for Class 1** as the primary model-selection metric.

---

## Dataset

Two datasets are used:

```text
Train.csv
Test.csv
```

### Training data

- **20,000 observations**
- **40 predictor variables**
- **1 target variable**
- Total columns: **41**

### Test data

- **5,000 observations**
- Same 40 predictors
- Target included for final evaluation

### Features

The sensor features are anonymized/ciphered:

```text
V1, V2, V3, ... V40
```

All predictor variables are continuous numerical features.

### Target

```text
Target
```

- `0` = No Failure
- `1` = Failure

---

## Exploratory Data Analysis

The notebook performs:

- Dataset shape and datatype inspection
- Statistical summaries
- Missing-value analysis
- Class-distribution analysis
- Histograms for all features
- Feature-to-target correlation analysis
- Pairplots for strongly correlated features
- Boxplots of important features by target class

### Key EDA Findings

- The training dataset contains **20,000 rows and 41 columns**.
- The test dataset contains **5,000 rows and 41 columns**.
- Missing values are sparse: only a small fraction of rows contain missing values.
- The target is **strongly imbalanced**:
  - Approximately **95.6%** No Failure
  - Approximately **4.4%** Failure
- Accuracy alone would therefore be misleading.
- No single variable has an extremely strong linear correlation with failure.
- Some of the stronger target relationships were observed for features such as `V18`, `V21`, `V15`, `V7`, `V16`, and `V39`.
- The combination of multiple features contains useful failure-detection signal, motivating a nonlinear neural-network approach.

---

## Data Preprocessing

The project avoids data leakage by fitting preprocessing objects only on the training split.

### Processing workflow

1. Separate predictors and target
2. Split training data into train and validation sets
3. Use a **stratified split** to preserve the failure ratio
4. Apply median imputation to missing numerical values
5. Fit the imputer only on training data
6. Standardize features using `StandardScaler`
7. Fit the scaler only on training data
8. Transform validation and test data using the training-fitted preprocessing objects
9. Convert feature arrays to `float32` for TensorFlow
10. Compute class weights from the training data for selected experiments

The training data is split into:

- Training: **16,000 observations**
- Validation: **4,000 observations**

The separate test dataset contains:

- Test: **5,000 observations**

---

## Evaluation Metric

### Primary Metric: F1-score for Class 1

The project uses **F1-score for the failure class** as the main model-selection metric.

Why?

- The dataset is highly imbalanced.
- Accuracy can appear excellent even when the model misses most failures.
- Recall is important because a False Negative can lead to generator replacement.
- Precision is also important because too many False Positives increase inspection costs.
- F1-score balances precision and recall.

### Supporting Metric

- ROC-AUC

Additional metrics include:

- Accuracy
- Precision
- Recall
- Confusion Matrix
- Classification Report

---

## Neural Network Experiments

Seven neural-network configurations are compared.

### Model 0 — Baseline SGD

- 1 hidden layer
- 64 units
- ReLU
- SGD optimizer
- No dropout
- No class weights

### Model 1 — Deeper SGD Network

- Additional hidden layers
- SGD optimizer

This configuration struggled with the imbalanced target and collapsed toward the majority class.

### Model 2 — Adam Optimizer

- Multi-layer neural network
- Adam optimizer

Adam substantially improved minority-class learning over SGD.

### Model 3 — Adam + Dropout

- 2 hidden layers
- Adam optimizer
- Dropout regularization

This configuration produced the strongest validation F1-score and was selected as the final model.

### Model 4 — Adam + Class Weights

Class weighting substantially increased recall but reduced precision, resulting in many false alarms.

### Model 5 — Deeper Network + Dropout + Adam

A deeper architecture with dropout was evaluated to test whether additional capacity improved failure detection.

### Model 6 — Deeper Network + Dropout + Adam + Class Weights

This model combined deeper layers, dropout, Adam, and class weighting.

It achieved high recall but comparatively low precision and F1-score due to excessive false positives.

---

## Model Comparison

The notebook compares all models using training and validation:

- F1-score
- Recall
- Precision
- ROC-AUC

The final model is selected using:

1. Highest validation F1-score
2. Strong ROC-AUC as a supporting metric
3. Generalization between training and validation performance

---

## Final Model

### Model 3 — Adam + Dropout

The final model achieved approximately:

### Validation Performance

- Precision: **98.5%**
- Recall: **90.5%**
- F1-score: **94.4%**
- ROC-AUC: **94.8%**

### Test Performance

- Accuracy: **99.0%**
- Precision: **98.0%**
- Recall: **84.8%**
- F1-score: **90.9%**
- ROC-AUC: **93.3%**

On the test set, the model correctly identified approximately **85% of actual failures** while maintaining very high precision.

---

## Business Interpretation

The selected model offers a useful balance between detecting failures and avoiding unnecessary inspections.

Compared with aggressive class-weighted models, Model 3 maintained:

- High failure recall
- Very high precision
- Lower false-alarm volume
- Strong overall F1-score

This is particularly useful where:

```text
Replacement Cost > Repair Cost > Inspection Cost
```

The model can therefore support maintenance teams by identifying turbines that should be inspected or repaired before complete failure.

---

## Key Business Recommendations

### 1. Use the model as a predictive-maintenance trigger

High-risk generators can be scheduled for inspection or repair before a breakdown occurs.

### 2. Tune the classification threshold based on real maintenance costs

The default threshold is `0.5`.

If generator replacement is significantly more expensive than inspection, lowering the classification threshold may increase recall and catch more failures.

### 3. Monitor model drift

Sensor behavior can change because of:

- Equipment aging
- Weather
- Seasonal operating conditions
- Changes in turbine hardware
- Sensor calibration

Track model performance over time and retrain when F1-score or recall deteriorates.

### 4. Prioritize False Negative monitoring

Missed failures are the most expensive errors. Production monitoring should explicitly track failure-class recall.

### 5. Combine predictions with maintenance operations

Predictions should be integrated with:

- Maintenance schedules
- Equipment history
- Technician inspections
- Replacement planning

The model should support engineering decisions rather than operate as an isolated automated decision system.

---

## Repository Structure

```text
renewind-predictive-maintenance/
├── README.md
├── ReneWind_Predictive_Maintenance_Neural_Network.ipynb
├── Train.csv
├── Test.csv
├── requirements.txt
└── .gitignore
```

---

## Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow
- Keras
- Jupyter Notebook
- Google Colab

---

# How to Run the Project

## Option 1 — Google Colab

### Step 1: Open Google Colab

Go to:

```text
https://colab.research.google.com/
```

### Step 2: Upload the notebook

Upload:

```text
ReneWind_Predictive_Maintenance_Neural_Network.ipynb
```

### Step 3: Upload the datasets

Upload both:

```text
Train.csv
Test.csv
```

The notebook currently loads:

```python
train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test.csv")
```

Therefore, place the CSV files in the same Colab working directory as the notebook session.

### Step 4: Install dependencies

The notebook includes a package installation cell.

You can also install them manually:

```python
!pip install tensorflow==2.19.0 scikit-learn==1.6.1 matplotlib==3.10.0 seaborn==0.13.2 numpy==2.0.2 pandas==2.2.2
```

If Colab asks for a runtime restart after package installation:

```text
Runtime → Restart session
```

Then continue from the imports/data-loading cell.

### Step 5: Run all cells

Select:

```text
Runtime → Run all
```

The notebook will:

1. Load the data
2. Perform EDA
3. Preprocess the data
4. Train the baseline neural network
5. Train six additional model configurations
6. Compare validation results
7. Select the best model
8. Evaluate the final model on the test set
9. Generate business recommendations

---

## Option 2 — Run Locally

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/renewind-predictive-maintenance.git
cd renewind-predictive-maintenance
```

### Step 2: Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
ReneWind_Predictive_Maintenance_Neural_Network.ipynb
```

and run all cells.

---

## Recommended Environment

```text
Python 3.10+
TensorFlow 2.19
```

For faster neural-network training, a GPU-enabled Google Colab runtime can be used:

```text
Runtime → Change runtime type → T4 GPU
```

---

## Important Notes

- The sensor variables are ciphered/anonymized, so domain interpretation of individual features is intentionally limited.
- Exact neural-network results may vary slightly due to stochastic optimization and differences in TensorFlow/GPU execution.
- The notebook uses the provided test set only for final-model evaluation.
- This project is intended for educational and portfolio purposes.

---

## Author

**Sreeman Mandava**

Machine Learning | Data Engineering | Software Engineering | AI Engineering
