import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Function to load and preprocess the dataset
def load_and_preprocess_data(filepath):
    # Loading the dataset from the specified file
    data = pd.read_csv(filepath)

    # Encoding categorical variables for better model processing
    label_encoder = LabelEncoder()
    data["drop_orientation"] = label_encoder.fit_transform(data["drop_orientation"])
    data["final_orientation"] = label_encoder.fit_transform(data["final_orientation"])

    return data


# Function to train SVM Classifier
def train_svm_classifier(X_train, y_train):
    # Initializing and training the SVM Classifier
    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)
    return svm_classifier


# Function to train and tune SVM Regressor
def train_and_tune_svm_regressor(X_train, y_train, param_grid):
    # Setting up GridSearch for hyperparameter tuning and training the regressor
    grid_search = GridSearchCV(SVR(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


# Function to plot predictions for regression tasks
def plot_predictions(y_test, y_pred, title, xlabel, ylabel):
    # Creating scatter plot for actual vs predicted values
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Load and preprocess the data
data = load_and_preprocess_data("data.csv")

# Splitting the dataset for classification and regression tasks
# For classification, we are predicting the 'final_orientation'
# For regression, we are predicting 'location_x' and 'location_y'

# Extracting relevant features for the classification task
X_classification = data[
    ["drop_number", "drop_orientation", "drop_angle_surface", "drop_polar_angle"]
]
y_orientation = data["final_orientation"]

# Feature engineering and preparing data for regression
interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_regression = interaction.fit_transform(
    data[["drop_number", "drop_orientation", "drop_angle_surface", "drop_polar_angle"]]
)
y_location_x = data["final_location_x"]
y_location_y = data["final_location_y"]

# Scaling the features to normalize the data, enhancing model performance
scaler_classification = StandardScaler()
X_classification_scaled = scaler_classification.fit_transform(X_classification)

scaler_regression = StandardScaler()
X_regression_scaled = scaler_regression.fit_transform(X_regression)

# Splitting the dataset into training and testing sets for both classification and regression
X_train_class, X_test_class, y_train_orientation, y_test_orientation = train_test_split(
    X_classification_scaled, y_orientation, test_size=0.15, random_state=1
)

# The regression task requires a different split due to different targets
(
    X_train_reg,
    X_test_reg,
    y_train_location_x,
    y_test_location_x,
    y_train_location_y,
    y_test_location_y,
) = train_test_split(
    X_regression_scaled, y_location_x, y_location_y, test_size=0.24, random_state=1
)

# Training the SVM classifier for the orientation classification task
svm_classifier = train_svm_classifier(X_train_class, y_train_orientation)

# Setting up hyperparameters for tuning the SVM regressors
param_grid_svr_x = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
param_grid_svr_y = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}

# Training and tuning the SVM regressors for predicting location x and y
best_svr_x = train_and_tune_svm_regressor(
    X_train_reg, y_train_location_x, param_grid_svr_x
)
best_svr_y = train_and_tune_svm_regressor(
    X_train_reg, y_train_location_y, param_grid_svr_y
)

# Evaluating the trained models on the test set and predictions from the classifier
y_pred_orientation_test = svm_classifier.predict(X_test_class)
accuracy_svm_orientation_test = accuracy_score(
    y_test_orientation, y_pred_orientation_test
)

# Predictions from the regressors
y_pred_location_x_test = best_svr_x.predict(X_test_reg)
y_pred_location_y_test = best_svr_y.predict(X_test_reg)

# Calculating Mean Squared Error for regression predictions
mse_x_svm_test = mean_squared_error(y_test_location_x, y_pred_location_x_test)
mse_y_svm_test = mean_squared_error(y_test_location_y, y_pred_location_y_test)

# Displaying the performance metrics
print("SVM Classifier Performance:")
print("Accuracy for Orientation Prediction:", accuracy_svm_orientation_test)
print("\nSVM Regressors Performance:")
print("Mean Squared Error for Location X Prediction:", mse_x_svm_test)
print("Mean Squared Error for Location Y Prediction:", mse_y_svm_test)

# Additional evaluation metrics for the classifier
print("\nDetailed Classification Report for SVM Classifier:")
print(classification_report(y_test_orientation, y_pred_orientation_test))

# Visualizing the confusion matrix for the classification results
ConfusionMatrixDisplay.from_predictions(y_test_orientation, y_pred_orientation_test)
plt.title("Confusion Matrix for SVM Classifier")
plt.show()

# Plotting the actual vs predicted values for both regression tasks
plot_predictions(
    y_test_location_x,
    y_pred_location_x_test,
    "Predicted vs Actual for Location X (SVR)",
    "Actual Location X",
    "Predicted Location X",
)
plot_predictions(
    y_test_location_y,
    y_pred_location_y_test,
    "Predicted vs Actual for Location Y (SVR)",
    "Actual Location Y",
    "Predicted Location Y",
)
