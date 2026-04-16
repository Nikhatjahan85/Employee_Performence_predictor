import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

data = pd.DataFrame({
    'age': np.random.randint(22, 60, n),
    'experience': np.random.randint(0, 20, n),
    'department': np.random.choice(['IT', 'HR', 'Sales'], n),
    'salary': np.random.randint(20000, 100000, n),
    'training_hours': np.random.randint(0, 50, n),
    'projects_completed': np.random.randint(1, 20, n),
    'attendance': np.random.uniform(0.7, 1.0, n)
})

def performance(row):
    score = (row['experience'] * 0.3 +
             row['training_hours'] * 0.2 +
             row['projects_completed'] * 0.3 +
             row['attendance'] * 10)

    if score > 25:
        return 'High'
    elif score > 15:
        return 'Medium'
    else:
        return 'Low'

data['performance'] = data.apply(performance, axis=1)

data.to_csv("data/employee_data.csv", index=False)

print("✅ Dataset created successfully!")