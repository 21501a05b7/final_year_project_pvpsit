# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

class ParkinsonsStageClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {
            'SVM': SVC(kernel='rbf', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB()
        }
        self.results = {}

    def load_and_analyze_data(self, filepath):
        """
        Load and perform initial analysis of the dataset
        """
        # Load the dataset
        self.df = pd.read_csv(filepath)
        
        # Print initial dataset shape
        print("\nInitial Dataset Shape:")
        print(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        
        # Add stage classification based on multiple features
        self.df['stage'] = self.classify_stages(self.df)
        
        # Print value counts to verify multiple classes
        print("\nStage Distribution:")
        print(self.df['stage'].value_counts())
        
        # Basic information about the dataset
        print("\nDataset Info:")
        print(self.df.info())
        
        # Check for missing values
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        # Print final dataset shape after processing
        print("\nFinal Dataset Shape after adding stages:")
        print(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        
        return self.df

    def classify_stages(self, df):
        """
        Classify Parkinson's disease stages based on multiple features
        """
        # Calculate composite score using multiple features
        df['composite_score'] = (
            df['MDVP:Jitter(%)'] * 2 +
            df['Shimmer:APQ3'] * 1.5 +
            df['NHR'] * 3 +
            df['RPDE'] * 2 +
            df['DFA'] * 1.5 +
            df['spread1'] * 2 +
            df['spread2'] * 1.5
        )
        
        # Define stage boundaries using percentiles to ensure multiple classes
        stage_bounds = [
            df['composite_score'].quantile(0.2),
            df['composite_score'].quantile(0.4),
            df['composite_score'].quantile(0.6),
            df['composite_score'].quantile(0.8)
        ]
        
        # Classify stages
        stages = []
        for score in df['composite_score']:
            if score <= stage_bounds[0]:
                stages.append(1)  # Early stage
            elif score <= stage_bounds[1]:
                stages.append(2)  # Moderate stage
            elif score <= stage_bounds[2]:
                stages.append(3)  # Advanced stage
            elif score <= stage_bounds[3]:
                stages.append(4)  # Severe stage
            else:
                stages.append(5)  # Very severe stage
        
        return stages

    def perform_eda(self):
        """
        Perform Exploratory Data Analysis
        """
        # Distribution of stages
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='stage')
        plt.title('Distribution of Parkinson\'s Disease Stages')
        plt.xlabel('Stage')
        plt.ylabel('Count')
        plt.show()
        
        # Feature distributions across stages
        features_to_analyze = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'Shimmer:APQ3', 'NHR', 'RPDE']
        
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(features_to_analyze, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(data=self.df, x='stage', y=feature)
            plt.title(f'{feature} by Stage')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[features_to_analyze + ['stage']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.show()

    def preprocess_data(self):
        """
        Preprocess the data for modeling
        """
        # Select features
        X = self.df.drop(['name', 'stage', 'status', 'composite_score'], axis=1)
        y = self.df['stage']
        
        # Print shapes of feature and target variables
        print("\nFeature Matrix Shape:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42, stratify=y)
        
        # Print shapes of training and testing sets
        print("\nTraining and Testing Set Shapes:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Print shapes after scaling
        print("\nScaled Data Shapes:")
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate multiple models
        """
        # Print shapes of input data
        print("\nInput Data Shapes for Model Training:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted Stage')
            plt.ylabel('Actual Stage')
            plt.show()
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'cv_scores': cv_scores,
                'report': report,
                'confusion_matrix': conf_matrix
            }
            
            print(f"\nResults for {name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print("\nClassification Report:")
            print(report)

    def compare_model_performances(self):
        """
        Compare performances of all models
        """
        accuracies = {name: results['accuracy'] for name, results in self.results.items()}
        cv_scores = {name: results['cv_scores'].mean() for name, results in self.results.items()}
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(accuracies))
        width = 0.35
        
        plt.bar(x - width/2, list(accuracies.values()), width, label='Test Accuracy')
        plt.bar(x + width/2, list(cv_scores.values()), width, label='CV Accuracy')
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x, accuracies.keys(), rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    # Initialize classifier
    classifier = ParkinsonsStageClassifier()
    
    # Load and analyze data
    df = classifier.load_and_analyze_data('parkinsons.data')
    
    # Perform EDA
    classifier.perform_eda()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = classifier.preprocess_data()
    
    # Train and evaluate models
    classifier.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Compare model performances
    classifier.compare_model_performances()

if __name__ == "__main__":
    main()