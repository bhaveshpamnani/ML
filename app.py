import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna('', inplace=True)

    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df

def hyperparameter_tuning(model, param_grid, X, y):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def evaluate_model(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    return {
        'accuracy': round(np.mean(acc_scores) * 100, 2),
        'precision': round(np.mean(prec_scores) * 100, 2),
        'recall': round(np.mean(rec_scores) * 100, 2),
        'f1_score': round(np.mean(f1_scores) * 100, 2)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file uploaded!"

    file = request.files['file']

    if file.filename == '':
        return "No file selected!"

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        X, y, processed_df = preprocess_data(df)

        param_grids = {
            'Logistic Regression': {'C': [0.1, 1, 10]},
            'Decision Tree': {'max_depth': [3, 5, 10]},
            'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
            'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
        }

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Support Vector Machine': SVC(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }

        results = []
        feature_importance = {}

        for model_name, model in models.items():
            if model_name in param_grids:
                model = hyperparameter_tuning(model, param_grids[model_name], X, y)

            metrics = evaluate_model(model, X, y)
            results.append({'model': model_name, **metrics})

            if hasattr(model, 'feature_importances_'):
                model.fit(X, y)
                feature_importance[model_name] = model.feature_importances_

        feature_names = processed_df.columns[:-1]

        dataset_info = {
            'shape': processed_df.shape,
            'missing_values': processed_df.isnull().sum().to_dict(),
            'data_types': processed_df.dtypes.apply(str).to_dict()
        }

        chart_data = {
            'models': [r['model'] for r in results],
            'accuracies': [r['accuracy'] for r in results]
        }

        return render_template('results.html',
                               results=results,
                               feature_importance=feature_importance,
                               feature_names=feature_names,
                               dataset_info=dataset_info,
                               chart_data=chart_data,
                               enumerate=enumerate)
    else:
        return "Invalid file type. Only CSV files are allowed!"

if __name__ == '__main__':
    app.run(debug=True)
