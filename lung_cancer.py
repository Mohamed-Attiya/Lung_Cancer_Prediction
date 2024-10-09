import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix

# Set the style for plots
plt.style.use('fivethirtyeight')

# Ignore warnings
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('D:\\AI\\archive (2).zip')

# Explore the data: First 5 rows, last 5 rows, and one random sample
print(data.head(5))  # First 5 rows
print("=====================================")
print(data.tail(5))  # Last 5 rows
print("=====================================")
print(data.sample())  # One random row
print("=====================================")

# Data description: shape, info, and statistics
print(data.shape)  # Number of rows and columns
print("=====================================")
print(data.info())  # Data types and non-null counts
print("=====================================")
print(data.describe())  # Statistical summary
print("=====================================")

# Count occurrences of each level in the 'Level' column
level_counts = data['Level'].value_counts().reset_index()
level_counts.columns = ['Level', 'Count']  # Rename columns to be more explicit

# Plot pie chart for levels
plt.figure(figsize=(15, 5))
plt.pie(x=level_counts['Count'], labels=level_counts['Level'], autopct='%.2f%%')  # Use the correct column names
plt.legend(level_counts['Level'])
plt.show()

# Calculate correlation for specific columns and plot heatmap
corr_columns = ['Air Pollution', 'Alcohol use', 'Dust Allergy', 'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking']
corr = data[corr_columns].corr()
plt.figure(figsize=(15, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Health Risks')
plt.show()

# Boxplot of Age distribution by Chronic Lung Disease
plt.figure(figsize=(15, 9))
sns.boxplot(data=data, x='chronic Lung Disease', y='Age', hue='Level')
plt.title('Age Distribution by Chronic Lung Disease')
plt.xlabel('Chronic Lung Disease Level')
plt.ylabel('Age')
plt.show()

# Countplot for Occupational Hazards by Chronic Lung Disease Level
plt.figure(figsize=(15, 5))
sns.countplot(data=data, x='OccuPational Hazards', hue='Level')
plt.title('Chronic Lung Disease Prevalence by Occupational Hazards')
plt.xlabel('Occupational Hazards')
plt.ylabel('Count')
plt.legend(title='Chronic Lung Disease')
plt.show()

# Create multiple subplots for different distributions
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
sns.histplot(data['Age'], bins=10, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Histogram of Age')
sns.histplot(data['Air Pollution'], bins=10, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Histogram of Air Pollution')
sns.histplot(data['Alcohol use'], bins=2, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Histogram of Alcohol Use')
sns.countplot(x='Gender', data=data, ax=axes[1, 1])
axes[1, 1].set_title('Count Plot of Gender')
sns.countplot(x='Dust Allergy', data=data, ax=axes[2, 0])
axes[2, 0].set_title('Count Plot of Dust Allergy')
sns.countplot(x='Smoking', data=data, ax=axes[2, 1])
axes[2, 1].set_title('Count Plot of Smoking')
plt.tight_layout()
plt.show()

# Drop unnecessary columns
cols_to_drop = ['index', 'Patient Id']
data.drop(columns=cols_to_drop, inplace=True)
print(data)
print("=====================================")

# Create a heatmap for numeric columns only
numeric_data = data.select_dtypes(include=[np.number])

# Check if there are numeric columns before plotting
if not numeric_data.empty:
    plt.figure(figsize=(20, 15))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='magma')
    plt.show()
else:
    print("No numeric columns to display in the heatmap.")

# ======================================================
# Calculating Mutual Information (Non-linear relationships)

# Encode the target variable 'Level' into numeric values
le = LabelEncoder()
data['Level_encoded'] = le.fit_transform(data['Level'])

# Select the features and target
X = data[corr_columns]  # Independent variables
y = data['Level_encoded']  # Target variable

# Check for missing values in the selected features
missing_data = X.isnull().sum()
print("Missing values in the features:\n",missing_data)
X = X.dropna(axis=1)  # Drop columns with missing values

# Calculate Mutual Information for classification
mi = mutual_info_classif(X, y, discrete_features=False)

# Create a DataFrame to display the results
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi})  # Use X.columns instead of corr_columns
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# Display the mutual information scores
print("Mutual Information Scores:")
print(mi_df)
print("===================================")

# Plot the mutual information scores
plt.figure(figsize=(10, 5))
sns.barplot(x='Mutual Information', y='Feature', data=mi_df, palette='viridis')
plt.title('Mutual Information between Features and Target (Level)')
plt.show()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Initialize the models
model_one = KNeighborsClassifier()
model_two = DecisionTreeClassifier()
model_three = RandomForestClassifier(n_estimators = 100,class_weight = 'balanced')
model_four = AdaBoostClassifier(algorithm='SAMME')

# Lists to store results for different models
columns = ['KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier']
result_one = []
result_two = []
result_three = []

# Function to train the model, make predictions, and display the results
def cal(model):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    pred = model.predict(X_test)
    
    # Calculate accuracy, recall, and F1 score
    accuracy = accuracy_score(pred, y_test)
    recall = recall_score(pred, y_test, average='macro')
    f1 = f1_score(pred, y_test, average='macro') 
    
    # Calculate the confusion matrix
    cm = confusion_matrix(pred, y_test)
    
    # Store results
    result_one.append(accuracy)
    result_two.append(recall)
    result_three.append(f1)
    
    # Display the confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title(f'Confusion Matrix for {model.__class__.__name__}')
    plt.show()
    
    # Print model evaluation metrics
    print(model)
    print("===================================")
    print(f"Accuracy: {accuracy:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")
    print("===================================")
    print("Confusion Matrix:\n", cm)
    print("===================================")
    
# Evaluate each model using the cal function
cal(model_one)
cal(model_two)
cal(model_three)
cal(model_four)

# Create a DataFrame to display results
data_frame = pd.DataFrame({
    "Algorithm": columns,
    "Accuracy": result_one,
    "Recall": result_two,
    "F1_Score": result_three
})

# Print the DataFrame with the results
print(data_frame)

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data_frame.Algorithm, result_one, label="Accuracy", marker='o')
ax.plot(data_frame.Algorithm, result_two, label="Recall", marker='o')
ax.plot(data_frame.Algorithm, result_three, label="F1_Score", marker='o')
ax.set_title("Model Performance Comparison")
ax.set_xlabel("Algorithm")
ax.set_ylabel("Score")
ax.legend()
plt.show()    