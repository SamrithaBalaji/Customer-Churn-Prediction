import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("churn.csv")

# Convert target column to numbers
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Convert text columns to numeric
data = pd.get_dummies(data)

# Separate features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------- NEW CUSTOMER DATA --------
# Must match feature columns
new_customer = pd.DataFrame([{
    'tenure': 2,
    'MonthlyCharges': 70,
    'TotalCharges': 140,
    'Contract_Month-to-month': 1,
    'Contract_One year': 0,
    'Contract_Two year': 0,
    'PaymentMethod_Bank transfer': 0,
    'PaymentMethod_Credit card': 0,
    'PaymentMethod_Electronic check': 1,
    'PaymentMethod_Mailed check': 0
}])

# Predict
prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Prediction: Customer WILL CHURN")
else:
    print("Prediction: Customer will NOT churn")
