import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Encode target
    le = LabelEncoder()
    df['performance'] = le.fit_transform(df['performance'])

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('performance', axis=1)
    y = df['performance']

    return X, y, le