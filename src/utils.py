import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from src.exception import CustomException



def get_cross_val_score_summary(model, X_train, y_train, cv=5, scoring='accuracy'):
        
        try:
            cross_val = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            print('-'*70)
            print(f'Cross validation Score Summary: #Folds:{cv}, Score:{scoring}')
            print('-'*70)
            print(pd.Series(cross_val).describe())

        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  



def get_classification_report(model, df_train, X_train, y_train, X_test, y_test):
    
    try: 
        # Fitting model
        model.fit(X_train, y_train)
        print('Fitting model completed!')
        
        # Generate model metrics
        print('-'*70)
        print(f'Classification Report: Model:{type(model).__name__}')
        print('-'*70)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, -1]
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        roc_auc = roc_auc_score(y_test, y_proba)

        precision_0 = precision_score(y_test, y_pred, pos_label=0)
        recall_0 = recall_score(y_test, y_pred, pos_label=0)
        f1_0 = f1_score(y_test, y_pred, pos_label=0)
        precision_1 = precision_score(y_test, y_pred, pos_label=1)
        recall_1 = recall_score(y_test, y_pred, pos_label=1)
        f1_1 = f1_score(y_test, y_pred, pos_label=1)
        
        metrics_arr = np.array([train_acc, test_acc, roc_auc, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1])
        
        
        print("Train accuracy:", train_acc)
        print("Test accuracy:", test_acc)
        print("ROC-AUC score: ", roc_auc)
        print('-'*70)
        print(classification_report(y_test, y_pred))
        print('-'*70)
        print('Confusion Matrix on Test Set:')
        print(confusion_matrix(y_test, y_pred))
        print('-'*70)
        
        
        # Get feature importances
        ser_feat_imp= pd.Series(dict(zip(df_train.columns, model.feature_importances_))).sort_values(ascending=False)
        print('Top-10 important features:')
        print(ser_feat_imp.iloc[:10])
        print('-'*70)
        
        # # Plot feature importances (Top-k)
        # k = 50
        # plt.figure(figsize=(20, 4))
        # plt.bar(ser_feat_imp.iloc[:k].index, ser_feat_imp.iloc[:k].values)
        # plt.xticks(rotation=90)
        # plt.show()
        
        return metrics_arr
    
    except Exception as e:
        custom_exception = CustomException(e, sys)
        print(custom_exception) 


def save_object(file_path, obj):
    try:

        # Creating a directory
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)