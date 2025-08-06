import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_explore_data():
    df = pd.read_csv('drug200.csv')
    print("Dataset Overview:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    return df

def preprocess_data(df):
    le = LabelEncoder()
    df_processed = df.copy()
    
    df_processed['Sex'] = le.fit_transform(df_processed['Sex'])
    df_processed['BP'] = le.fit_transform(df_processed['BP'])
    df_processed['Cholesterol'] = le.fit_transform(df_processed['Cholesterol'])
    df_processed['Drug'] = le.fit_transform(df_processed['Drug'])
    
    print("\nProcessed Dataset:")
    print(df_processed.head())
    
    return df_processed

def train_model(df):
    X = df.drop('Drug', axis=1)
    y = df['Drug']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test, X

def visualize_tree(model, feature_names):
    """Create and display decision tree visualization"""
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, 
              class_names=["DrugA", "DrugB", "DrugC", "DrugX", "DrugY"], 
              filled=True)
    plt.title("Decision Tree for Drug Classification", fontsize=16)
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():
    df = load_and_explore_data()
    
    df_processed = preprocess_data(df)
    
    model, X_train, X_test, y_train, y_test, X = train_model(df_processed)
    
    visualize_tree(model, X.columns)
    
    evaluate_model(model, X_test, y_test)

main()
