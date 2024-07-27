import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.components.data_ingestion import DataIngest
from src.components.data_cleaning import DataCleaning


class FeatureEngineeringNonNLP:

    def __init__(self):
        self.expensive_item_price_threshold = -100
        self.psc_list = []
        self.pssc_list = []



    def temporal_feats(self, df_inp):
        
        try: 
            df = df_inp.copy()
            # 'project_submitted_datetime'
            df['project_submitted_datetime'] = pd.to_datetime(df['project_submitted_datetime'])
            df['sub_year'] = df['project_submitted_datetime'].dt.year
            df['sub_month'] = df['project_submitted_datetime'].dt.month
            df['sub_dow'] = df['project_submitted_datetime'].dt.dayofweek
            df['sub_hour'] = df['project_submitted_datetime'].dt.hour
            df.drop('project_submitted_datetime', axis=1, inplace=True)
            return df
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)


    def location_feats(self, df_inp):
        try:
            df = df_inp.copy()
            # 'school_state' : No Change
            df['school_state'] = df['school_state']
            return df
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)


    def teacher_feats(self, df_inp):
        try:
            df = df_inp.copy()
            # 'teacher_id'
            df['records_per_user'] = df.groupby(['teacher_id'])['id'].transform('count')
            df['records_per_user_cat'] = df['records_per_user'].apply(lambda x: 'Single Record' if x==1 else 'Multiple Record')
            df.drop('teacher_id', axis=1, inplace=True)
            
            # 'teacher_prefix'
            df['teacher_prefix'] = df['teacher_prefix'].apply(lambda x: 'Others' if x not in ['Mrs.', 'Ms.', 'Mr.'] else x)
            
            # 'teacher_number_of_previously_posted_projects' : No Change
            df['teacher_number_of_previously_posted_projects'] = df['teacher_number_of_previously_posted_projects']

            return df
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)



    def proj_cat_feats(self, df_inp, fit=0):

        try:
            # project_subject_categories
            df_psc = df_inp[['id', 'project_subject_categories']].copy()
            
            df_psc['psc'] = df_psc['project_subject_categories'].str.split(', ')
            df_psc_explode = df_psc.explode('psc')
            
            df_psc_clean = ~df_psc_explode.pivot(index='id', columns='psc').isna()
            df_psc_clean.columns = [tup[-1] for tup in df_psc_clean.columns]

            if fit==1:
                self.psc_list = df_psc_clean.columns

            if fit==0:
                if len(self.psc_list) == 0:
                    raise Exception('Object not fitted yet!')
                
                # If subject_category category present in test data and NOT in train data, simply ignore

                # If subject_category category present in train data and NOT in test data, add the subject category with all zeros
                for col in self.psc_list:
                    if col not in df_psc_clean.columns:
                        # print(f'{col} Not found in test data but present in train data')
                        df_psc_clean[col] = np.zeros(df_psc_clean.shape[0])
            
            df_psc_clean = df_psc_clean.astype('int64')
            
            if 'Warmth' in df_psc_clean.columns:
                df_psc_clean.drop('Warmth', axis=1, inplace=True)
            if 'Care & Hunger' in df_psc_clean.columns:
                df_psc_clean.rename(columns={'Care & Hunger':'Warmth, Care & Hunger'}, inplace=True)

            return df_psc_clean
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)



    def proj_subcat_feats(self, df_inp, fit=0):

        try:
            # project_subject_subcategories
            df_pssc = df_inp[['id', 'project_subject_subcategories']].copy()
            
            df_pssc['pssc'] = df_pssc['project_subject_subcategories'].str.split(', ')
            df_pssc_explode = df_pssc.explode('pssc')
            
            df_pssc_clean = ~df_pssc_explode.pivot(index='id', columns='pssc').isna()
            df_pssc_clean.columns = [tup[-1] for tup in df_pssc_clean.columns]

            if fit==1:
                self.pssc_list = df_pssc_clean.columns

            if fit==0:
                if len(self.pssc_list) == 0:
                    raise Exception('Object not fitted yet!')
                
                # If subject_category category present in test data and NOT in train data, simply ignore

                # If subject_category category present in train data and NOT in test data, add the subject category with all zeros
                for col in self.pssc_list:
                    if col not in df_pssc_clean.columns:
                        # print(f'{col} Not found in test data but present in train data')
                        df_pssc_clean[col] = np.zeros(df_pssc_clean.shape[0])  
        


            df_pssc_clean = df_pssc_clean.astype('int64')

            if 'Warmth' in df_pssc_clean.columns:
                df_pssc_clean.drop('Warmth', axis=1, inplace=True)
            if 'Care & Hunger' in df_pssc_clean.columns:
                df_pssc_clean.rename(columns={'Care & Hunger':'Warmth, Care & Hunger'}, inplace=True)

            
            return df_pssc_clean
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)


    def resource_feats_fit(self, df_inp):

        try:
            df_res = df_inp.copy()
            # Finding upper whisker of price distribution (log scale)
            item_price_log_iqr = df_res['item_price_log'].quantile(0.75)-df_res['item_price_log'].quantile(0.25)
            item_price_log_upp_whis = df_res['item_price_log'].quantile(0.75) + (1.5*item_price_log_iqr)
            self.expensive_item_price_threshold = item_price_log_upp_whis


        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)

    def resource_feats_transform(self, df_inp):

        try:

            df_res = df_inp.copy()

            # Tagging an item as expensive if it's price is above upper whisker of the price distribution (log) :
            if self.expensive_item_price_threshold == -100:
                raise Exception('Object not fitted yet!')
            df_res['expensive_item_quantity'] = df_res[['quantity', 'item_price_log']].apply(lambda x: x.iloc[0] if x.iloc[1]>self.expensive_item_price_threshold else 0, axis=1)


            # Finding total price per item (price of 1 item X quantity)
            df_res['total_item_price'] = df_res['quantity']*df_res['price']
            

            # Aggregating df_res on project id
            df_res_agg = df_res.groupby('id')[['quantity', 'expensive_item_quantity','description','total_item_price']].agg({'quantity':['sum'], 
                                                                                                                            'expensive_item_quantity':['sum'], 
                                                                                                                            'description':['nunique'], 
                                                                                                                            'total_item_price' : ['sum']})
            
            df_res_agg.columns = [('_').join(tup) for tup in df_res_agg.columns]
            df_res_agg.reset_index(inplace=True)
            df_res_agg.rename(columns={'quantity_sum':'res_item_q', 'expensive_item_quantity_sum': 'res_exp_item_q', 
                                       'description_nunique': 'res_item_uniq_q', 'total_item_price_sum': 'res_price'}, inplace=True)
            
            
            
            # Binning res_exp_item_q
            df_res_agg['res_exp_item_qcat'] = pd.cut(df_res_agg['res_exp_item_q'], bins=[0, 1, 4, 100], labels=['Zero', 'Upto3', 'Above3'], right=False).astype('str')
            
            # Applying log transformation on res_item_q, res_item_uniq_q, res_price
            df_res_agg['res_item_q_ln'] = np.log(df_res_agg['res_item_q'])
            df_res_agg['res_item_uniq_q_ln'] = np.log(df_res_agg['res_item_uniq_q'])
            df_res_agg['res_price_ln'] = np.log(df_res_agg['res_price'])
            
            
            df_res_agg_clean = df_res_agg[['id', 'res_item_q_ln', 'res_item_uniq_q_ln', 'res_exp_item_qcat', 'res_price_ln']].copy()

            return df_res_agg_clean


        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)



    def gen_non_nlp_feats(self, df_inp1:pd.DataFrame, df_inp2:pd.DataFrame, fit=0, test_mode=0):

        '''
        This function generates the below NON-NLP features:

        Temporal Features: 
        - 'sub_year'                                     : Submission Year
        - 'sub_month'                                    : Submission Month
        - 'sub_dow'                                      : Submission Day of Week
        - 'sub_hour'                                     : Submission Hour
        
        Location Features: 
        - 'school_state'                                 : State of the school from where submission is made
        
        Teacher Features: 
        - 'records_per_user'                             : Number of submissions made by the teacher (including the current submission)
        - 'records_per_user_cat'                         : 'First-Time' or 'Returning' teacher
        - 'teacher_prefix'                               : Teacher's Prefix ('Ms.', 'Mrs.', 'Mr.', 'Others')
        - 'teacher_number_of_previously_posted_projects' : Number of past submissions made by the teacher at the time of current submission
        
        Project Features:
        - 'project_grade_category'                       : Class Standard for whom the submission is made
        - 'project_subject_categories'                   : Subject Category of project 
        - 'project_subject_subcategories'                : Subject Sub-Category of project
        
        Resource Features:
        - 'res_item_q_ln'                                : Total quantity of all project resources requested for sponsorship (Log-Transformed)
        - 'res_item_uniq_ln'                             : Total unique project resources requested for sponsorship (Log-Transformed)
        - 'res_exp_item_qcat'                            : Total quantity of all expensive project resources requested for sponsorship (Categorical)
        - 'res_price_ln'                                 : Total price of all project resources requested for sponsorship (Log-Transformed)
        
        Inputs: 
        - Main dataframe
        - Resources dataframe (Aggregation at project level performed inside this function)
        - fit: 0/1

        Actions:
        - fit=1 : To be used with train data, fits the object first by learning parameters from the train data and uses it to perform feature engineering on the train data
        - fit=0 : To be used with test data, using the learnt parameters from train data, performs feature engineering on the test data
        
        '''

        
        try:
            if test_mode==1:
                df_inp1 = df_inp1.iloc[:1000].copy()
                df_inp2 = df_inp2.loc[df_inp2['id'].isin(df_inp1['id'].values)].copy()
            
            df = df_inp1[['id', 'project_is_approved', 
                          'teacher_id', 'teacher_prefix', 'school_state', 'project_submitted_datetime', 
                          'project_grade_category', 'project_subject_categories', 'project_subject_subcategories', 
                          'teacher_number_of_previously_posted_projects']].copy()
            df_res = df_inp2.copy()


            df = self.temporal_feats(df_inp=df)
            df = self.location_feats(df_inp=df)
            df = self.teacher_feats(df_inp=df)

            df_psc = self.proj_cat_feats(df_inp=df, fit=fit)
            df_pssc = self.proj_subcat_feats(df_inp=df, fit=fit)


            # Merging categories and sub-categories
            df_proj_cat_merged = pd.merge(df_psc, df_pssc, left_index=True, right_index=True, how='left').reset_index()
            if 'Warmth, Care & Hunger_y' in df_proj_cat_merged.columns :

                df_proj_cat_merged.drop(['Warmth, Care & Hunger_y'], axis=1, inplace=True)
                df_proj_cat_merged.rename({'Warmth, Care & Hunger_x':'Warmth, Care & Hunger'}, axis=1, inplace=True)

            if 'Special Needs_y' in df_proj_cat_merged.columns:
                df_proj_cat_merged.drop(['Special Needs_y'], axis=1, inplace=True)
                df_proj_cat_merged.rename({'Special Needs_x':'Special Needs'}, axis=1, inplace=True)

            
            df.drop('project_subject_categories', axis=1, inplace=True)
            df.drop('project_subject_subcategories', axis=1, inplace=True)

            
            df_res['item_price_log'] = np.log(df_res['price'])

            if fit==1:
                # Learn Parameters from df_res
                self.resource_feats_fit(df_inp=df_res)
            # Using learnt parameters perform feature engineering
            df_res_agg = self.resource_feats_transform(df_inp=df_res)

            # Merging all the dfs together
            df_non_nlp = pd.merge(df, df_proj_cat_merged, on='id', how='left')
            df_non_nlp = pd.merge(df_non_nlp, df_res_agg, on='id', how='left')

            return df_non_nlp

        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)   




if __name__=='__main__':

    print('Feature Engineering NON-NLP testing started')

    # print_sep_len = 100

    # ingest_obj = DataIngest()
    # train_pt, test_pt, train_res_pt, test_res_pt = ingest_obj.start_data_ingestion_from_csv(sample_size=0.15, test_size=0.25, random_state=42)

    # train_df = pd.read_csv(train_pt)
    # test_df = pd.read_csv(test_pt)
    # train_res_df = pd.read_csv(train_res_pt)
    # test_res_df = pd.read_csv(test_res_pt)

    # print(train_df.shape, test_df.shape)
    # print(train_res_df.shape, test_res_df.shape)
    # print('-'*print_sep_len)


    # data_clean_obj = DataCleaning()

    # train_non_text_df = data_clean_obj.clean_nontext_feat(train_df)
    # test_non_text_df = data_clean_obj.clean_nontext_feat(test_df)
    # print('Non-Text DF:')
    # print(train_non_text_df.shape, test_non_text_df.shape)
    # print(train_non_text_df.columns)
    # print('-'*print_sep_len)

    # train_text_df = data_clean_obj.clean_text_feat(train_df)
    # test_text_df = data_clean_obj.clean_text_feat(test_df)
    # print('Text DF:')
    # print(train_text_df.shape, test_text_df.shape)
    # print(train_text_df.columns)
    # print('-'*print_sep_len)

    # train_res_df = data_clean_obj.clean_res(df_inp=train_res_df, fit=1)
    # test_res_df = data_clean_obj.clean_res(df_inp=test_res_df, fit=0)
    # print('Resource DF:')
    # print(train_res_df.shape, test_res_df.shape)
    # print(train_res_df.columns)
    # print('-'*print_sep_len)


    # fe_non_nlp_obj = FeatureEngineeringNonNLP()
    # train_non_nlp_df = fe_non_nlp_obj.gen_non_nlp_feats(df_inp1=train_non_text_df, df_inp2=train_res_df, fit=1, test_mode=0)
    # test_non_nlp_df = fe_non_nlp_obj.gen_non_nlp_feats(df_inp1=test_non_text_df, df_inp2=test_res_df, fit=0, test_mode=0)
    # print('Non-NLP DF:')
    # print(train_non_nlp_df.shape)
    # print(train_non_nlp_df.columns)
    # print(train_non_nlp_df.isna().sum())
    # print('-'*print_sep_len)
    # print(test_non_nlp_df.shape)
    # print(test_non_nlp_df.columns)
    # print(test_non_nlp_df.isna().sum())
    






    
            











