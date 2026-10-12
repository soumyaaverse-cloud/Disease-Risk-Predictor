import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import os

df = pd.read_csv("data/diabetes.csv")
print(df.head())
print(df["Outcome"].value_counts())

# these columns have 0s that are actually missing data
fakeZeroCols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in fakeZeroCols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

df["BmiCategory"] = pd.cut(
    df["BMI"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)
df = pd.get_dummies(df, columns=["BmiCategory"], drop_first=True)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

xTrain, xTest, yTrain, yTest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(xTrain, yTrain)

yPred = model.predict(xTest)
acc = accuracy_score(yTest, yPred)
print(f"test accuracy: {acc*100:.2f}%")

cm = confusion_matrix(yTest, yPred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("model/confusion_matrix.png")
plt.show()

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh", figsize=(8, 6))
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("model/feature_importance.png")
plt.show()

os.makedirs("model", exist_ok=True)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("model saved")