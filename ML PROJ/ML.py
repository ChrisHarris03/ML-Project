import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

# Load the dataset (replace 'your_dataset.csv' with your actual file)
data = pd.read_csv('salary_dataset.csv')

# Extract features (X) and target variable (y)
X = data[['YearsExperience', 'EducationLevel', 'Position']]
y = data['Salary']

# Encode categorical variables using LabelEncoder and OneHotEncoder
label_encoder = LabelEncoder()
X['Position'] = label_encoder.fit_transform(X['Position'])

# One-hot encode the 'EducationLevel' column
X = pd.get_dummies(X, columns=['EducationLevel', 'Position'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot actual vs predicted salaries
plt.scatter(y_test, y_pred, color='blue', label='Original Data')

# User input for new data point
new_experience = (input("Enter Years of Experience: "))
new_education = input("Enter Education Level (High School, Bachelor, Master, PhD): ")
new_position = input("Enter Position (Junior, Mid, Senior, Manager): ")

# Transform user input
new_position_encoded = label_encoder.transform([new_position])
new_education_encoded = pd.get_dummies([new_education], columns=['EducationLevel'], drop_first=True)

# Ensure the lengths match before creating the DataFrame
if len(new_experience) == len(new_education_encoded):
    # Create a DataFrame for the new data point
    new_data = pd.DataFrame({
        'YearsExperience': [new_experience],
        'EducationLevel': new_education_encoded.iloc[0],
        'Position': new_position_encoded[0]
    })

    # Predict the salary for the new data point
    new_salary_prediction = model.predict(new_data)

    # Add the new data point to the plot
    plt.scatter([new_salary_prediction], [new_salary_prediction], color='red', marker='X', label='New Prediction')
    plt.xlabel("Actual Salaries")
    plt.ylabel("Predicted Salaries")
    plt.title("Actual vs Predicted Salaries")
    plt.legend()
    plt.show()
else:
    print("Error: Length mismatch. Please ensure all arrays have the same length.")
