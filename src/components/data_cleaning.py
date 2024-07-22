
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.components.data_ingestion import DataIngest


class DataCleaning:

    def __init__(self):
        self.miss_des_dict = {}
        self.fit_flag = 0


    def clean_nontext_feat(self, df_inp):
        '''
        This function does the following tasks:
        - Drops unnecessary column 'Unnamed: 0'
        - Drops those rows where 'teacher_prefix' is missing
        - Drops duplicate rows
        - Removes the textual features
        '''

        try:
            df_proj_app = df_inp.copy()

            # Drop column 'Unnamed: 0'
            df_proj_app.drop('Unnamed: 0', axis=1, inplace=True)

            # Drop rows with missing values in 'teacher_prefix'
            df_proj_app.dropna(subset=['teacher_prefix'], axis=0, inplace=True)
        
            # Drop Duplicates
            df_proj_app.drop_duplicates(inplace=True)

            # Removing the textual features
            df_proj_app.drop(['project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', 'project_resource_summary'], axis=1, inplace=True)

            return df_proj_app
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)
    

    def clean_text_feat(self, df_inp):
        '''
        This function does the following tasks:
        - Seperates out the textual features 
        - Fixes the missing values in the essay features
        - Before 17th May, 2016: Combine old1 + old2 as new1 and old3 + old4 as new2
        - From 17th May, 2016: new1 = old1 and new2 = old2
        '''

        try: 
            df_text_feat = df_inp[['id', 'project_title', 'project_resource_summary', 'project_is_approved']].copy()

            # Fixing the missing values in project essays
            # Fixing the missing values in project essays: Step1
            df_ess = df_inp[['id', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', 'project_submitted_datetime']].copy()
            df_ess['dos'] = pd.to_datetime(df_ess['project_submitted_datetime']).dt.date.astype('str')
        
            # Fixing the missing values in project essays: Step2 (Handling the case on & after 17th May, 2016)
            df_ess['proj_essay1'] = df_ess['project_essay_1']
            df_ess['proj_essay2'] = df_ess['project_essay_2']

            # Fixing the missing values in project essays: Step3 (Handling the case before 17th May, 2016)
            df_ess.loc[df_ess['dos']<'2016-05-17', 'proj_essay1'] = df_ess.loc[df_ess['dos']<'2016-05-17', 'project_essay_1'] + \
                                                                    ' ' + \
                                                                    df_ess.loc[df_ess['dos']<'2016-05-17', 'project_essay_2']

            df_ess.loc[df_ess['dos']<'2016-05-17', 'proj_essay2'] = df_ess.loc[df_ess['dos']<'2016-05-17', 'project_essay_3'] + \
                                                                    ' ' + \
                                                                    df_ess.loc[df_ess['dos']<'2016-05-17', 'project_essay_4'] 
            
            df_ess_no_missval = df_ess[['id', 'proj_essay1', 'proj_essay2']].copy()

            # Merging all the textual features after handling missing values
            df_text_feat = pd.merge(df_text_feat, df_ess_no_missval, on='id', how='inner')

            return df_text_feat
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)


    def impute_res_description(self, price):
        try:
            if price in self.miss_des_dict:
                return self.miss_des_dict[price]
            else:
                return 'Unknown'
            
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)



    def clean_res(self, df_inp, fit=0):
        '''
        This function does the following tasks:
        - If the price of a resource is less than $1, cap it to $1
        - Drops duplicate rows

        Actions:
        - fit=1: To be used with train data, for the missing values in 'description', learns the mapping b/w price and most frequently occuring resource for the same price
        - fit=0: To be used with test data, using learnt mapping, imputes the missing values in 'description' with the most frequently occuring resource for the same price
        '''

        try: 
            df_res = df_inp.copy()

            # If price is less than $1, then cap it to $1
            df_res.loc[df_res['price']<1, 'price'] = 1
                
            # Drop Duplicates
            df_res.drop_duplicates(inplace=True)
            
            # Fixing missing values in 'description'

            if fit==1:

                self.fit_flag=1

                # Step1: Finding the price of those rows where description is missing
                miss_des_unique_price = df_res.loc[df_res['description'].isna(), 'price'].sort_values().unique()
                # print(miss_des_unique_price)

                # Step2: Implementing the strategy for imputation in code: Creating a dictionary which stores the mapping b/w price and most frequent item
                self.miss_des_dict = {}
                df_res_with_des = df_res.loc[~df_res['description'].isna()]
                price_range = 2.5

                for price in miss_des_unique_price:
                    df_match = df_res_with_des.loc[(df_res_with_des['price']>=(price-price_range)) & (df_res_with_des['price']<=(price+price_range))]
                    # print(price, df_match.shape[0])
                    if df_match.shape[0]>0:
                        mode_val = df_match['description'].mode()[0]
                        self.miss_des_dict[price] = mode_val

            
            # Step3: Imputing missing values in 'description'
            if self.fit_flag==0:
                raise Exception('Object not fitted yet!')
            # df_res['description'] = df_res[['description', 'price']].apply(lambda x: self.impute_res_description(x[1]) if x[0]=='' or (isinstance(x[0], float) and np.isnan(x[0])) else x[0], axis=1)   
            # df_res['description'] = df_res[['description', 'price']].apply(lambda x: self.miss_des_dict[x[1]] if pd.isna(x[0]) else x[0], axis=1)  
            df_res['description'] = df_res[['description', 'price']].apply(lambda x: self.impute_res_description(x.iloc[1]) if pd.isna(x.iloc[0]) else x.iloc[0], axis=1)   

            return  df_res
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)




    

if __name__=='__main__':

    print('Data Cleaning testing started')

    print_sep_len = 100
    ingest_obj = DataIngest()
    train_pt, test_pt, train_res_pt, test_res_pt = ingest_obj.start_data_ingestion_from_csv(sample_size=1, test_size=0.25, random_state=42)

    # train_df = pd.read_csv(train_pt)
    # test_df = pd.read_csv(test_pt)
    # print(train_df.shape, test_df.shape)
    

    # # Test Methods: clean_nontext_feat, clean_text_feat
    # print('Show all columns in dataframe:')
    # print(train_df.columns)
    # print('-'*print_sep_len)
    # print('Show missing columns:')
    # print(train_df.isna().sum())
    # print('-'*print_sep_len)

    # data_clean_obj = DataCleaning()
    # train_non_text_df = data_clean_obj.clean_nontext_feat(train_df)
    # train_text_df = data_clean_obj.clean_text_feat(train_df)

    # print('Show all columns in dataframe:')
    # print(train_non_text_df.columns)
    # print('-'*print_sep_len)
    # print('Show missing columns:')
    # print(train_non_text_df.isna().sum())
    # print('-'*print_sep_len)


    # print('Show all columns in dataframe:')
    # print(train_text_df.columns)
    # print('-'*print_sep_len)
    # print('Show missing columns:')
    # print(train_text_df.isna().sum())
    # print('-'*print_sep_len)


    train_res_df = pd.read_csv(train_res_pt)
    test_res_df = pd.read_csv(test_res_pt)
    print(train_res_df.shape, test_res_df.shape)


    # Test Methods: clean_res_fit_transform, clean_res_transform
    print('Show all columns in dataframe:')
    print(train_res_df.columns)
    print('-'*print_sep_len)
    print('Show missing columns:')
    print(train_res_df.isna().sum())
    print('-'*print_sep_len)


    print('Show all columns in dataframe:')
    print(test_res_df.columns)
    print('-'*print_sep_len)
    print('Show missing columns:')
    print(test_res_df.isna().sum())
    print('-'*print_sep_len)

    data_clean_obj = DataCleaning()
    train_res_df = data_clean_obj.clean_res(df_inp=train_res_df, fit=1)
    test_res_df = data_clean_obj.clean_res(df_inp=test_res_df, fit=0)


    print('Show all columns in dataframe:')
    print(train_res_df.columns)
    print('-'*print_sep_len)
    print('Show missing columns:')
    print(train_res_df.isna().sum())
    print('-'*print_sep_len)


    print('Show all columns in dataframe:')
    print(test_res_df.columns)
    print('-'*print_sep_len)
    print('Show missing columns:')
    print(test_res_df.isna().sum())
    print('-'*print_sep_len)


    print(train_res_df['price'].min(), test_res_df['price'].min())










        



    
