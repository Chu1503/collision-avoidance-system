import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump, load  # Import joblib to save the model

# Load data from CSV files
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# Assuming that the feature columns are the same for both train and test data
X_train = df_train.iloc[:, [0, 1, 2, 3]].values
y_train = df_train.iloc[:, [4]].values
X_test = df_test.iloc[:, [0, 1, 2, 3]].values
y_test = df_test.iloc[:, [4]].values

# Initialize and train the model
model = RandomForestRegressor(max_depth=7, n_estimators=20)
model.fit(X_train, y_train.ravel())  # Flatten y_train to 1D array for compatibility

# Make predictions
y_predicted = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
score = model.score(X_test, y_test)

# Print performance metrics
print("RandomForestRegressor Results:")
print("R^2 score = %.2f%%" % (score * 100))
print("MAE = %.2f" % mae)
print("MSE = %.2f" % mse)
print("="*50)

# Save the model as a .pkl file using joblib
dump(model, 'models/random_forest_model.joblib')
print("Model saved as random_forest_model.joblib")

# Optional: Prepare the results for exporting to CSV
results = pd.DataFrame({
    'Actual': y_test.ravel(),
    'Predicted': y_predicted
})
results.to_csv('data/predictions.csv', index=False)
print("Predictions exported to predictions.csv")