import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

df = pd.read_csv('data/raw/iris.csv')

le = LabelEncoder()
df['target'] = le.fit_transform(df['species'])

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.joblib')

os.makedirs('data/processed', exist_ok=True)
pd.concat([X_train, y_train], axis=1).to_csv('data/processed/train.csv', index=False)
pd.concat([X_test, y_test], axis=1).to_csv('data/processed/eval.csv', index=False)

print("✅ Training complete.")
