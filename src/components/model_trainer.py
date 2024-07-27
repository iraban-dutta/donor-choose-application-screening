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

        self.fitted_tar_enc_path = []
        self.fitted_lab_enc_path = []
        self.fitted_feat_scaler_path = os.path.join('artifacts', 'scaler_features.pkl')
        self.trained_model_path = os.path.join('artifacts', 'model.pkl')


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
            X_train_c2n, X_test_c2n, tar_dict, lab_dict = data_preprocess_obj4.cat_to_num_transform(X_train=X_train, y_train=y_train, 
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


            # Saving cat_2_num encoders as pickle file
            for col in tar_dict:
                self.fitted_tar_enc_path.append(os.path.join('artifacts', f'enc_target_{col}.pkl'))

            for col in lab_dict:
                self.fitted_lab_enc_path.append(os.path.join('artifacts', f'enc_label_{col}.pkl'))

            for i, col in enumerate(tar_dict.keys()):
                save_object(file_path=self.fitted_tar_enc_path[i], obj=tar_dict[col])

            for i, col in enumerate(lab_dict.keys()):
                save_object(file_path=self.fitted_lab_enc_path[i], obj=lab_dict[col])

            print('Cat2Num Encoders Saved')
            print('-'*print_sep_len)


            print('Cat2Num Transformation end')
            print('-'*print_sep_len)



            # Data Preprocessing: Feature Scaling
            print('Feature Scaling start')
            print('-'*print_sep_len)

            X_train_scl, X_test_scl, feat_scaler = data_preprocess_obj4.standard_scaler(X_train=X_train_c2n, X_test=X_test_c2n)
            print(X_train_scl.shape, X_test_scl.shape)

            # Saving feature scaler as pickle file
            save_object(file_path=self.fitted_feat_scaler_path, obj=feat_scaler)
            print('Feature Scaler Saved')
            print('-'*print_sep_len)

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


        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  
