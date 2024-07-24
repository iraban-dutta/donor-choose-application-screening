import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from src.exception import CustomException
from src.components.data_preprocessing import DataPreProcessing
from src.utils import get_cross_val_score_summary, get_classification_report, save_object, load_object




class ModelTrainer:

    def __init__(self):

        self.trained_model_path = os.path.join('artifacts', 'model.pkl')



    # def get_cross_val_score_summary(self, model, X_train, y_train, cv=5, scoring='accuracy'):
        
    #     try:
    #         cross_val = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    #         print('-'*70)
    #         print(f'Cross validation Score Summary: #Folds:{cv}, Score:{scoring}')
    #         print('-'*70)
    #         print(pd.Series(cross_val).describe())

    #     except Exception as e:
    #         custom_exception = CustomException(e, sys)
    #         print(custom_exception)  


    # def get_classification_report(self, model, df_train, X_train, y_train, X_test, y_test):
        
    #     try: 
    #         # Fitting model
    #         model.fit(X_train, y_train)
    #         print('Fitting model completed!')
            
    #         # Generate model metrics
    #         print('-'*70)
    #         print(f'Classification Report: Model:{type(model).__name__}')
    #         print('-'*70)
    #         y_pred = model.predict(X_test)
    #         y_proba = model.predict_proba(X_test)[:, -1]
    #         train_acc = model.score(X_train, y_train)
    #         test_acc = model.score(X_test, y_test)
    #         roc_auc = roc_auc_score(y_test, y_proba)

    #         precision_0 = precision_score(y_test, y_pred, pos_label=0)
    #         recall_0 = recall_score(y_test, y_pred, pos_label=0)
    #         f1_0 = f1_score(y_test, y_pred, pos_label=0)
    #         precision_1 = precision_score(y_test, y_pred, pos_label=1)
    #         recall_1 = recall_score(y_test, y_pred, pos_label=1)
    #         f1_1 = f1_score(y_test, y_pred, pos_label=1)
            
    #         metrics_arr = np.array([train_acc, test_acc, roc_auc, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1])
            
            
    #         print("Train accuracy:", train_acc)
    #         print("Test accuracy:", test_acc)
    #         print("ROC-AUC score: ", roc_auc)
    #         print('-'*50)
    #         print(classification_report(y_test, y_pred))
    #         print('-'*50)
    #         print('Confusion Matrix on Test Set:')
    #         print(confusion_matrix(y_test, y_pred))
    #         print('-'*50)
            
            
    #         # Get feature importances
    #         ser_feat_imp= pd.Series(dict(zip(df_train.columns, model.feature_importances_))).sort_values(ascending=False)
    #         print('Top-10 important features:')
    #         print(ser_feat_imp.iloc[:10])
    #         print('-'*50)
            
    #         # # Plot feature importances (Top-k)
    #         # k = 50
    #         # plt.figure(figsize=(20, 4))
    #         # plt.bar(ser_feat_imp.iloc[:k].index, ser_feat_imp.iloc[:k].values)
    #         # plt.xticks(rotation=90)
    #         # plt.show()
            
    #         return metrics_arr
        
    #     except Exception as e:
    #         custom_exception = CustomException(e, sys)
    #         print(custom_exception) 




    def train_model(self, train_df:pd.DataFrame, test_df:pd.DataFrame, sample_size=1, read_presaved_data=0):

        '''
        This method will:
        - Preprocess data
        - Train model
        - Evaluate model

        Inputs:
        - sample_size: Sample size to train model on

        '''

        try: 
            
            print_sep_len = 100

            if read_presaved_data==1:
                # Data Preprocessing: Split into target and independent variables
                train_final_df = pd.read_csv('artifacts/data_post_feat_eng/train_final.csv')
                test_final_df = pd.read_csv('artifacts/data_post_feat_eng/test_final.csv')
            else:
                train_final_df = train_df
                test_final_df = test_df


            print(train_final_df.shape, test_final_df.shape)
            print('-'*print_sep_len)

            if train_final_df.shape[1]!=test_final_df.shape[1]:
                print('Column Mismatch b/w train and test data')
                for col in train_final_df.columns:
                    if col not in test_final_df.columns:
                        print(col)
                print('-'*print_sep_len)


            print('Missing values:')
            print(train_final_df.isna().sum().loc[train_final_df.isna().sum()>0])
            print(test_final_df.isna().sum().loc[test_final_df.isna().sum()>0])
            print('-'*print_sep_len)


            train_final_df_sample = train_final_df.sample(frac=sample_size, random_state=42)
            test_final_df_sample = test_final_df.sample(frac=sample_size, random_state=42)

            X_train = train_final_df_sample.drop(['id', 'project_is_approved'], axis=1).copy()
            X_test = test_final_df_sample.drop(['id', 'project_is_approved'], axis=1).copy()
            y_train = train_final_df_sample['project_is_approved']
            y_test = test_final_df_sample['project_is_approved']
            print(X_train.shape, y_train.shape)
            print(X_test.shape, y_test.shape)
            print('-'*print_sep_len)


            # Data Preprocessing: Categorical to Numerical Transformation
            print('Cat2Num Transformation start')
            print('-'*print_sep_len)

            data_preprocess_obj4 = DataPreProcessing()
            cat_cols, num_cols, other_cols = data_preprocess_obj4.split_cols_cat_num(X_train)
            print(cat_cols)

            tar_enc_cols = ['teacher_prefix', 'school_state', 'project_grade_category', 'res_exp_item_qcat']
            lab_enc_cols = ['records_per_user_cat']
            X_train_c2n, X_test_c2n = data_preprocess_obj4.cat_to_num_transform(X_train=X_train, y_train=y_train, 
                                                                                X_test=X_test, y_test=y_test, 
                                                                                tar_cols=tar_enc_cols, 
                                                                                lab_cols=lab_enc_cols)
            
            # cat_cols, num_cols, other_cols = data_preprocess_obj4.split_cols_cat_num(X_train_c2n)
            # print('Post Cat2Num Transformation')
            # print(cat_cols)
            # print('-'*print_sep_len)

            # cat_cols, num_cols, other_cols = data_preprocess_obj4.split_cols_cat_num(X_test_c2n)
            # print('Post Cat2Num Transformation')
            # print(cat_cols)
            # print('-'*print_sep_len)

            print('Cat2Num Transformation end')
            print('-'*print_sep_len)


            # Data Preprocessing: Feature Scaling
            print('Feature Scaling start')
            print('-'*print_sep_len)

            X_train_scl, X_test_scl = data_preprocess_obj4.standard_scaler(X_train=X_train_c2n, X_test=X_test_c2n)
            print(X_train_scl.shape, X_test_scl.shape)

            print('Feature Scaling end')
            print('-'*print_sep_len)
     
            # Model Training & Evaluation:
            # xgbc = xgb.XGBClassifier()

            xgbc = xgb.XGBClassifier(objective= 'binary:logistic', 
                                     n_estimators=400, 
                                     max_depth=4, 
                                     learning_rate=0.1, 
                                     gamma=0.25, 
                                     subsample=0.7, 
                                     colsample_bytree=0.5, 
                                     random_state=42)

            # get_cross_val_score_summary(model=xgbc, X_train=X_train_scl, y_train=y_train, cv=5, scoring='accuracy')

            xgbc_metrics_arr = get_classification_report(model=xgbc, df_train=X_train, 
                                                         X_train=X_train_scl, y_train=y_train, 
                                                         X_test=X_test_scl, y_test=y_test)
            

            # Saving model as pickle file
            save_object(file_path=self.trained_model_path, obj=xgbc)
            print('Model Saved')
            print('-'*print_sep_len)

            
            # Loading model frm pickle file
            trained_model = load_object(file_path='artifacts/model.pkl')
            print('Model Loaded')
            print('-'*print_sep_len)

            print(X_test_scl.shape)
            print(X_test_scl[0].reshape(1, -1).shape)

            print('True Label:', y_test[0])
            print('Prediction:', xgbc.predict(X_test_scl[0].reshape(1, -1)))
            print('Prediction with saved model:', trained_model.predict(X_test_scl[0].reshape(1, -1)))



        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  
