# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# # Load dataset
# df = pd.read_csv("dataset.csv")

# # Drop unnecessary columns
# drop_cols = ['Case_No', 'Who completed the test']
# df = df.drop(columns=drop_cols)

# # Encode categorical values
# label_encoder = LabelEncoder()

# for column in df.columns:
#     if df[column].dtype == 'object':
#         df[column] = label_encoder.fit_transform(df[column])

# # Split data
# X = df.drop('Class/ASD Traits ', axis=1)
# y = df['Class/ASD Traits ']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train interpretable model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Evaluate
# pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, pred))
# print(classification_report(y_test, pred))

# # Save model
# joblib.dump(model, "model.pkl")

# print("Model saved as model.pkl")import pandas as pd
# --> 2nd fix
# import pandas as pd
# import joblib


# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# # Load dataset
# df = pd.read_csv("dataset.csv")

# # Select only the screening questions
# X = df[['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']]

# # Target variable
# y = df['Class/ASD Traits ']

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Evaluate
# pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, pred))
# print(classification_report(y_test, pred))

# # Save model
# joblib.dump(model, "model.pkl")

# print("Model saved as model.pkl")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("dataset.csv")

# Autism questionnaire columns
features = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']

X = df[features]

# Target column
target = 'Class/ASD Traits '

# Convert labels clearly
# Yes = ASD (1), No = Non-ASD (0)

df[target] = df[target].map({
    "Yes":1,
    "No":0
})

y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save model
joblib.dump(model, "model.pkl")

print("Model saved successfully")