import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate logistic regression model."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = np.array(model.predict_proba(X_test))[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    return model, metrics

def train_knn(X_train, y_train, X_test, y_test):
    """Train and evaluate KNN model."""
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = np.array(model.predict_proba(X_test))[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    return model, metrics

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train and evaluate decision tree model."""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = np.array(model.predict_proba(X_test))[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    return model, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate random forest model."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = np.array(model.predict_proba(X_test))[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    return model, metrics

def train_svm(X_train, y_train, X_test, y_test):
    """Train and evaluate SVM model."""
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = np.array(model.predict_proba(X_test))[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    return model, metrics

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate model evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    metrics['roc_auc'] = auc(fpr, tpr)
    
    return metrics
