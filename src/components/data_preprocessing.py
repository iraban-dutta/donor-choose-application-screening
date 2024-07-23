import sys
import os
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException



class DataPreProcessing:

    def __init__(self):
        pass


    def merge_nlp_basic_feats(self, df_title:pd.DataFrame, df_ess1:pd.DataFrame, df_ess2:pd.DataFrame, df_ess_sim:pd.DataFrame, df_res_sum:pd.DataFrame):

        try:
            df_merge1_ess = pd.merge(df_ess1, df_ess2, on='id', how='inner')
            df_merge2_ess = pd.merge(df_merge1_ess, df_ess_sim, on='id', how='inner')
            df_merge3 = pd.merge(df_title, df_merge2_ess, on='id', how='inner')
            df_merge4 = pd.merge(df_merge3, df_res_sum, on='id', how='inner')

            return df_merge4

        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  


    def prep_nlp_basic_df(self, df_inp:pd.DataFrame):
        try:
            df_nlp_basic = df_inp.drop(['project_title_cln', 'proj_essay1_cln', 'proj_essay2_cln', 'project_resource_summary_cln'], axis=1).copy()
            return df_nlp_basic
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)              


    def prep_nlp_w2v_df(self, df_inp:pd.DataFrame):
        try:
            df_text_clean = df_inp[['id', 'project_title_cln', 'proj_essay1_cln', 'proj_essay2_cln', 'project_resource_summary_cln']].copy()
            # df_text_clean.fillna('', inplace=True)
            return df_text_clean
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)     


    def merge_nlp_w2v_feats(self, df_title:pd.DataFrame, df_ess1:pd.DataFrame, df_ess2:pd.DataFrame, df_res_sum:pd.DataFrame):

        try:
            df_merge1 = pd.merge(df_ess1, df_ess2, left_index=True, right_index=True, how='inner')
            df_merge2 = pd.merge(df_title, df_merge1, left_index=True, right_index=True, how='inner')
            df_merge3 = pd.merge(df_merge2, df_res_sum, left_index=True, right_index=True, how='inner').reset_index()

            return df_merge3

        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)    



    def merge_all_feats(self, df_non_nlp:pd.DataFrame, df_nlp_basic:pd.DataFrame, df_nlp_w2v:pd.DataFrame):

        try: 
            df_merge1 = pd.merge(df_non_nlp, df_nlp_basic, on='id', how='inner')
            df_merge2 = pd.merge(df_merge1, df_nlp_w2v, on='id', how='inner')

            return df_merge2

        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)



    def split_cols_cat_num(self, df:pd.DataFrame):
        
        try:
            cat_cols = []
            num_cols = []
            other_cols = []

            for col in df.columns:
                col_dtype = df[col].dtype
                if col_dtype=='object':
                    cat_cols.append(col)
                elif pd.api.types.is_integer_dtype(col_dtype) or pd.api.types.is_float_dtype(col_dtype):
                # elif df[col].dtype=='int' or df[col].dtype=='float':
                    num_cols.append(col)
                else:
                    other_cols.append(col)

            print(f'Categorical columns length: {len(cat_cols)}')
            print(f'Numerical columns length: {len(num_cols)}')
            print(f'Other columns length: {len(other_cols)}')

            # print(cat_cols)
            # print('-'*50)
            # print(other_cols)
            # print('-'*50)
            # print(num_cols)
            # print('-'*50)

        
            return cat_cols, num_cols, other_cols

        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)


    def cat_to_num_transform(self, X_train:pd.DataFrame, y_train:pd.Series, X_test:pd.DataFrame, y_test:pd.Series, tar_cols:list, lab_cols:list):

        try:
            df_train = X_train.copy()
            df_test = X_test.copy()

            tar_enc_dict = {}
            for col in tar_cols:
                tar_enc_dict[col] = TargetEncoder()
                df_train[col] = tar_enc_dict[col].fit_transform(df_train[col], y_train)
                df_test[col] = tar_enc_dict[col].transform(df_test[col], y_test)
                

            lab_enc_dict = {}
            for col in lab_cols:
                lab_enc_dict[col] = LabelEncoder()
                df_train[col] = lab_enc_dict[col].fit_transform(df_train[col])
                df_test[col] = lab_enc_dict[col].transform(df_test[col])

            return df_train, df_test
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)




    def standard_scaler(self, X_train:pd.DataFrame, X_test:pd.DataFrame):

        std_scaler = StandardScaler()
        X_train_scl = std_scaler.fit_transform(X_train)
        X_test_scl = std_scaler.transform(X_test)

        return X_train_scl, X_test_scl



        


        

