# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your dataset
data = pd.read_csv(r'C:\Users\vidisha\OneDrive\Desktop\VS Code\VS CODE\ic cream recommendation\example-2.csv')  # Replace 'your_dataset.csv' with the path to your dataset

# Preprocess the data
# You may need to encode categorical variables such as Gender and Zodiac sign using one-hot encoding.

# Define your features and target variable
X = data[[ 'Age','Gender', 'Which ice cream topping do you enjoy the most?', 'Whats ur zodiac sign?']]
y = data['Favourite ice cream flavour?']

# Perform one-hot encoding for categorical variables


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=300, random_state=1)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# You can save the trained model for future use
# import joblib
# joblib.dump(clf, 'ice_cream_model.pkl')



import joblib

# Train your model (replace with your actual model training code)

# Save the model to a file
joblib.dump(clf, 'recommendation_model.pkl')