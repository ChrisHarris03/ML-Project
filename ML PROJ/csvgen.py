import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 1000

experience = np.random.uniform(1, 20, num_samples)
education_levels = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], num_samples)
positions = np.random.choice(['Junior', 'Mid', 'Senior', 'Manager'], num_samples)

# Define coefficients for linear regression
experience_coeff = 1000
education_coeff = {'High School': 500, 'Bachelor': 1000, 'Master': 1500, 'PhD': 2000}
position_coeff = {'Junior': -500, 'Mid': 1000, 'Senior': 2000, 'Manager': 3000}

# Calculate salary based on linear regression equation
salary = (experience_coeff * experience +
          np.vectorize(education_coeff.get)(education_levels) +
          np.vectorize(position_coeff.get)(positions) +
          np.random.normal(0, 1000, num_samples))

# Create a DataFrame
data = pd.DataFrame({
    'YearsExperience': experience,
    'EducationLevel': education_levels,
    'Position': positions,
    'Salary': salary
})

# Save the dataset to a CSV file
data.to_csv('salary_dataset.csv', index=False)

# Display the first few rows of the generated dataset
print(data.head())
