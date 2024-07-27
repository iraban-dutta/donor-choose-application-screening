import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.components.data_ingestion import DataIngest
from src.components.data_cleaning import DataCleaning
from src.components.feat_eng_non_nlp import FeatureEngineeringNonNLP
from src.components.feat_eng_nlp_basic1 import FeatureEngineeringNLPBasic1
from src.components.feat_eng_nlp_basic2 import FeatureEngineeringNLPBasic2
from src.components.feat_eng_nlp_basic3 import FeatureEngineeringNLPBasic3
from src.components.feat_eng_nlp_w2v import FeatureEngineeringNLPW2V
from src.components.data_preprocessing import DataPreProcessing
from src.components.model_trainer import ModelTrainer
from src.utils import save_object, load_object




class TrainPipeline:

    def __init__(self, sample_n=5000, sample_size :float=None, test_size=0.25, random_state=42):

        self.sample_n = sample_n
        self.sample_size = sample_size
        self.test_size = test_size
        self.random_state = random_state

        self.train_non_nlp_path = os.path.join('artifacts', 'data_post_feat_eng', 'train_non_nlp.csv')
        self.test_non_nlp_path = os.path.join('artifacts', 'data_post_feat_eng', 'test_non_nlp.csv')

        # self.train_nlp_basic_title_path = os.path.join('artifacts', 'data_post_feat_eng', 'train_nlp_basic_title.csv')
        # self.test_nlp_basic_title_path = os.path.join('artifacts', 'data_post_feat_eng', 'test_nlp_basic_title.csv')

        # self.train_nlp_basic_ess1_path = os.path.join('artifacts', 'data_post_feat_eng', 'train_nlp_basic_ess1.csv')
        # self.test_nlp_basic_ess1_path = os.path.join('artifacts', 'data_post_feat_eng', 'test_nlp_basic_ess1.csv')

        # self.train_nlp_basic_ess2_path = os.path.join('artifacts', 'data_post_feat_eng', 'train_nlp_basic_ess2.csv')
        # self.test_nlp_basic_ess2_path = os.path.join('artifacts', 'data_post_feat_eng', 'test_nlp_basic_ess2.csv')

        # self.train_nlp_basic_ess_sim_path = os.path.join('artifacts', 'data_post_feat_eng', 'train_nlp_basic_ess_sim.csv')
        # self.test_nlp_basic_ess_sim_path = os.path.join('artifacts', 'data_post_feat_eng', 'test_nlp_basic_ess_sim.csv')

        # self.train_nlp_basic_res_sum_path = os.path.join('artifacts', 'data_post_feat_eng', 'train_nlp_basic_res_sum.csv')
        # self.test_nlp_basic_res_sum_path = os.path.join('artifacts', 'data_post_feat_eng', 'test_nlp_basic_res_sum.csv')

        self.train_nlp_basic_path = os.path.join('artifacts', 'data_post_feat_eng', 'train_nlp_basic.csv')
        self.test_nlp_basic_path = os.path.join('artifacts', 'data_post_feat_eng', 'test_nlp_basic.csv')

        self.train_nlp_w2v_path = os.path.join('artifacts', 'data_post_feat_eng', 'train_nlp_w2v.csv')
        self.test_nlp_w2v_path = os.path.join('artifacts', 'data_post_feat_eng', 'test_nlp_w2v.csv')

        self.train_final_df_path = os.path.join('artifacts', 'data_post_feat_eng', 'train_final.csv')
        self.test_final_df_path = os.path.join('artifacts', 'data_post_feat_eng', 'test_final.csv')


        self.non_nlp_obj_path = os.path.join('artifacts', 'non_nlp_obj.pkl') 

        self.nlp_basic_obj_title_path = os.path.join('artifacts', 'nlp_basic_obj_title.pkl')
        self.nlp_basic_obj_ess1_path = os.path.join('artifacts', 'nlp_basic_obj_ess1.pkl')
        self.nlp_basic_obj_ess2_path = os.path.join('artifacts', 'nlp_basic_obj_ess2.pkl')
        self.nlp_basic_obj_ess_sim_path = os.path.join('artifacts', 'nlp_basic_obj_ess_sim.pkl')
        self.nlp_basic_obj_res_sum_path = os.path.join('artifacts', 'nlp_basic_obj_res_sum.pkl')

        self.gensim_w2v_title_path = os.path.join('artifacts', 'gensim_w2v_title.pkl')
        self.gensim_w2v_ess1_path = os.path.join('artifacts', 'gensim_w2v_ess1.pkl')
        self.gensim_w2v_ess2_path = os.path.join('artifacts', 'gensim_w2v_ess2.pkl')
        self.gensim_w2v_res_sum_path = os.path.join('artifacts', 'gensim_w2v_res_sum.pkl')




    def train_model_from_scratch(self, save_data_post_feat_eng=0, model_train=1):

        '''
        This method will:
        - Ingest data
        - Clean data
        - Extract features
        - Preprocess data
        - Train model
        - Evaluate model

        Inputs:
        - save_data_post_feat_eng: 0/1

        Actions:
        - save_data_post_feat_eng=1 : Create a copy of the preprocessed data for future use
        - save_data_post_feat_eng=0 : No copy made of preprocessed data
        '''

        try:
            print_sep_len = 100

            # Ingest Data
            print('Data Ingestion start')
            print('-'*print_sep_len)

            ingest_obj = DataIngest()
            train_pt, test_pt, train_res_pt, test_res_pt = ingest_obj.start_data_ingestion_from_csv(sample_n=self.sample_n, 
                                                                                                    sample_size=self.sample_size, 
                                                                                                    test_size=self.test_size, 
                                                                                                    random_state=self.random_state)
            

            train_df = pd.read_csv(train_pt)
            test_df = pd.read_csv(test_pt)
            train_res_df = pd.read_csv(train_res_pt)
            test_res_df = pd.read_csv(test_res_pt)
            
            print('Data Ingestion end')
            print('-'*print_sep_len)



            # Data Cleaning
            print('Data Cleaning start')
            print('-'*print_sep_len)

            data_clean_obj = DataCleaning()

            train_non_text_df = data_clean_obj.clean_nontext_feat(df_inp=train_df)
            test_non_text_df = data_clean_obj.clean_nontext_feat(df_inp=test_df)

            train_text_df = data_clean_obj.clean_text_feat(df_inp=train_df)
            test_text_df = data_clean_obj.clean_text_feat(df_inp=test_df)

            train_res_df = data_clean_obj.clean_res(df_inp=train_res_df, fit=1)
            test_res_df = data_clean_obj.clean_res(df_inp=test_res_df, fit=0)

            print('Data Cleaning end')
            print('-'*print_sep_len)



            # Generate Features : Non NLP
            print('Feature Generation: Non-NLP start')
            print('-'*print_sep_len)

            fe_non_nlp_obj = FeatureEngineeringNonNLP()
            train_non_nlp_df = fe_non_nlp_obj.gen_non_nlp_feats(df_inp1=train_non_text_df, df_inp2=train_res_df, fit=1, test_mode=0)
            test_non_nlp_df = fe_non_nlp_obj.gen_non_nlp_feats(df_inp1=test_non_text_df, df_inp2=test_res_df, fit=0, test_mode=0)

            print('Feature Generation: Non-NLP end')
            print('-'*print_sep_len)



            # Generate Features : NLP-Basic
            print('Feature Generation: NLP-Basic start')
            print('-'*print_sep_len)

            fe_nlp_basic_title_obj = FeatureEngineeringNLPBasic1()
            train_title_df = fe_nlp_basic_title_obj.gen_nlp_basic_title_feats(df_inp=train_text_df, fit=1, test_mode=0)
            test_title_df = fe_nlp_basic_title_obj.gen_nlp_basic_title_feats(df_inp=test_text_df, fit=0, test_mode=0)

            fe_nlp_basic_ess1_obj = FeatureEngineeringNLPBasic2()
            train_ess1_df = fe_nlp_basic_ess1_obj.gen_nlp_basic_ess_feats(df_inp=train_text_df, col='proj_essay1', fit=1, test_mode=0)
            test_ess1_df = fe_nlp_basic_ess1_obj.gen_nlp_basic_ess_feats(df_inp=test_text_df, col='proj_essay1', fit=0, test_mode=0)

            fe_nlp_basic_ess2_obj = FeatureEngineeringNLPBasic2()
            train_ess2_df = fe_nlp_basic_ess2_obj.gen_nlp_basic_ess_feats(df_inp=train_text_df, col='proj_essay2', fit=1, test_mode=0)
            test_ess2_df = fe_nlp_basic_ess2_obj.gen_nlp_basic_ess_feats(df_inp=test_text_df, col='proj_essay2', fit=0, test_mode=0)

            fe_nlp_basic_ess_sim_obj = FeatureEngineeringNLPBasic2()
            train_ess_sim_df = fe_nlp_basic_ess_sim_obj.gen_nlp_basic_ess_similar_feats(df_ess1=train_ess1_df, df_ess2=train_ess2_df, test_mode=0)
            test_ess_sim_df = fe_nlp_basic_ess_sim_obj.gen_nlp_basic_ess_similar_feats(df_ess1=test_ess1_df, df_ess2=test_ess2_df, test_mode=0)

            fe_nlp_basic_res_sum_obj = FeatureEngineeringNLPBasic3()
            train_res_sum_df = fe_nlp_basic_res_sum_obj.gen_nlp_basic_res_sum_feats(df_inp=train_text_df, fit=1, test_mode=0)
            test_res_sum_df = fe_nlp_basic_res_sum_obj.gen_nlp_basic_res_sum_feats(df_inp=test_text_df, fit=0, test_mode=0)


            data_preprocess_obj1 = DataPreProcessing()
            train_merge_nlp_basic_df = data_preprocess_obj1.merge_nlp_basic_feats(df_title=train_title_df, 
                                                                                df_ess1=train_ess1_df, 
                                                                                df_ess2=train_ess2_df, 
                                                                                df_ess_sim=train_ess_sim_df, 
                                                                                df_res_sum=train_res_sum_df)
            
            test_merge_nlp_basic_df = data_preprocess_obj1.merge_nlp_basic_feats(df_title=test_title_df, 
                                                                                df_ess1=test_ess1_df, 
                                                                                df_ess2=test_ess2_df, 
                                                                                df_ess_sim=test_ess_sim_df, 
                                                                                df_res_sum=test_res_sum_df)
            
            train_nlp_basic_df = data_preprocess_obj1.prep_nlp_basic_df(df_inp=train_merge_nlp_basic_df)
            test_nlp_basic_df = data_preprocess_obj1.prep_nlp_basic_df(df_inp=test_merge_nlp_basic_df)
            
            print('Feature Generation: NLP-Basic end')
            print('-'*print_sep_len)



            # Generate Features : NLP-W2V
            print('Feature Generation: NLP-W2V start')
            print('-'*print_sep_len)

            data_preprocess_obj2 = DataPreProcessing()
            train_nlp_w2v_inp_df = data_preprocess_obj2.prep_nlp_w2v_df(df_inp=train_merge_nlp_basic_df)
            test_nlp_w2v_inp_df = data_preprocess_obj2.prep_nlp_w2v_df(df_inp=test_merge_nlp_basic_df)

            # Some samples in the textual features (post cleaning) can become missing (due to stopword removal)
            train_nlp_w2v_inp_df.fillna('', inplace=True)
            test_nlp_w2v_inp_df.fillna('', inplace=True)


            fe_nlp_w2v_title_obj = FeatureEngineeringNLPW2V()
            fe_nlp_w2v_title_obj.train_gensim_model(df=train_nlp_w2v_inp_df, col='project_title_cln', vec_size=100)
            train_nlp_w2v_title_df = fe_nlp_w2v_title_obj.generate_doc_avg_w2v_df(df=train_nlp_w2v_inp_df, col='project_title_cln', prefix='title', sk_flag=1)
            test_nlp_w2v_title_df = fe_nlp_w2v_title_obj.generate_doc_avg_w2v_df(df=test_nlp_w2v_inp_df, col='project_title_cln', prefix='title', sk_flag=1)
            
            fe_nlp_w2v_ess1_obj = FeatureEngineeringNLPW2V()
            fe_nlp_w2v_ess1_obj.train_gensim_model(df=train_nlp_w2v_inp_df, col='proj_essay1_cln', vec_size=100)
            train_nlp_w2v_ess1_df = fe_nlp_w2v_ess1_obj.generate_doc_avg_w2v_df(df=train_nlp_w2v_inp_df, col='proj_essay1_cln', prefix='ess1', sk_flag=1)
            test_nlp_w2v_ess1_df = fe_nlp_w2v_ess1_obj.generate_doc_avg_w2v_df(df=test_nlp_w2v_inp_df, col='proj_essay1_cln', prefix='ess1', sk_flag=1)

            fe_nlp_w2v_ess2_obj = FeatureEngineeringNLPW2V()
            fe_nlp_w2v_ess2_obj.train_gensim_model(df=train_nlp_w2v_inp_df, col='proj_essay2_cln', vec_size=100)
            train_nlp_w2v_ess2_df = fe_nlp_w2v_ess2_obj.generate_doc_avg_w2v_df(df=train_nlp_w2v_inp_df, col='proj_essay2_cln', prefix='ess2', sk_flag=1)
            test_nlp_w2v_ess2_df = fe_nlp_w2v_ess2_obj.generate_doc_avg_w2v_df(df=test_nlp_w2v_inp_df, col='proj_essay2_cln', prefix='ess2', sk_flag=1)

            fe_nlp_w2v_res_sum_obj = FeatureEngineeringNLPW2V()
            fe_nlp_w2v_res_sum_obj.train_gensim_model(df=train_nlp_w2v_inp_df, col='project_resource_summary_cln', vec_size=100)
            train_nlp_w2v_res_sum_df = fe_nlp_w2v_res_sum_obj.generate_doc_avg_w2v_df(df=train_nlp_w2v_inp_df, col='project_resource_summary_cln', prefix='res_sum', sk_flag=1)
            test_nlp_w2v_res_sum_df = fe_nlp_w2v_res_sum_obj.generate_doc_avg_w2v_df(df=test_nlp_w2v_inp_df, col='project_resource_summary_cln', prefix='res_sum', sk_flag=1)


            train_nlp_w2v_df = data_preprocess_obj2.merge_nlp_w2v_feats(df_title=train_nlp_w2v_title_df, df_ess1=train_nlp_w2v_ess1_df, 
                                                                        df_ess2=train_nlp_w2v_ess2_df, df_res_sum=train_nlp_w2v_res_sum_df)
            test_nlp_w2v_df = data_preprocess_obj2.merge_nlp_w2v_feats(df_title=test_nlp_w2v_title_df, df_ess1=test_nlp_w2v_ess1_df, 
                                                                       df_ess2=test_nlp_w2v_ess2_df, df_res_sum=test_nlp_w2v_res_sum_df)
            train_nlp_w2v_df.fillna(0, inplace=True)
            test_nlp_w2v_df.fillna(0, inplace=True)


            print('Feature Generation: NLP-W2V end')
            print('-'*print_sep_len)


            # Stacking all features together
            data_preprocess_obj3 = DataPreProcessing()

            train_final_df = data_preprocess_obj3.merge_all_feats(df_non_nlp=train_non_nlp_df, df_nlp_basic=train_nlp_basic_df, df_nlp_w2v=train_nlp_w2v_df)
            test_final_df = data_preprocess_obj3.merge_all_feats(df_non_nlp=test_non_nlp_df, df_nlp_basic=test_nlp_basic_df, df_nlp_w2v=test_nlp_w2v_df)


            # # Sanity checks on data
            # print(train_non_nlp_df.shape)
            # print(train_non_nlp_df.columns)
            # print('-'*print_sep_len)
            # print(test_non_nlp_df.shape)
            # print(test_non_nlp_df.columns)
            # print('-'*print_sep_len)
            # print(train_nlp_basic_df.shape)
            # print(train_nlp_basic_df.columns)
            # print('-'*print_sep_len)
            # print(test_nlp_basic_df.shape)
            # print(test_nlp_basic_df.columns)
            # print('-'*print_sep_len)
            # print(train_nlp_w2v_df.shape)
            # print(train_nlp_w2v_df.columns)
            # print('-'*print_sep_len)
            # print(test_nlp_w2v_df.shape)
            # print(test_nlp_w2v_df.columns)
            # print('-'*print_sep_len)
            # print(train_final_df.shape)
            # print(train_final_df.columns)
            # print('-'*print_sep_len)
            # print(test_final_df.shape)
            # print(test_final_df.columns)
            # print('-'*print_sep_len)

            # print('Missing values:')
            # print(train_non_nlp_df.isna().sum().loc[train_non_nlp_df.isna().sum()>0])
            # print(test_non_nlp_df.isna().sum().loc[test_non_nlp_df.isna().sum()>0])
            # print(train_nlp_basic_df.isna().sum().loc[train_nlp_basic_df.isna().sum()>0])
            # print(test_nlp_basic_df.isna().sum().loc[test_nlp_basic_df.isna().sum()>0])
            # print(train_nlp_w2v_df.isna().sum().loc[train_nlp_w2v_df.isna().sum()>0])
            # print(test_nlp_w2v_df.isna().sum().loc[test_nlp_w2v_df.isna().sum()>0])
            # print(train_final_df.isna().sum().loc[train_final_df.isna().sum()>0])
            # print(test_final_df.isna().sum().loc[test_final_df.isna().sum()>0])
            # print('-'*print_sep_len)


            # Save fitted objects for feature engineering
            save_object(file_path=self.non_nlp_obj_path, obj=fe_non_nlp_obj)

            save_object(file_path=self.nlp_basic_obj_title_path, obj=fe_nlp_basic_title_obj)
            save_object(file_path=self.nlp_basic_obj_ess1_path, obj=fe_nlp_basic_ess1_obj)
            save_object(file_path=self.nlp_basic_obj_ess2_path, obj=fe_nlp_basic_ess2_obj)
            save_object(file_path=self.nlp_basic_obj_res_sum_path, obj=fe_nlp_basic_res_sum_obj)
        
            save_object(file_path=self.gensim_w2v_title_path, obj=fe_nlp_w2v_title_obj)
            save_object(file_path=self.gensim_w2v_ess1_path, obj=fe_nlp_w2v_ess1_obj)
            save_object(file_path=self.gensim_w2v_ess2_path, obj=fe_nlp_w2v_ess2_obj)
            save_object(file_path=self.gensim_w2v_res_sum_path, obj=fe_nlp_w2v_res_sum_obj)



            # # Testing saved objects for preprocessing: 
            # ld_obj1 = load_object(file_path='artifacts/non_nlp_obj.pkl')
            # ld_obj2 = load_object(file_path='artifacts/nlp_basic_obj_title.pkl')
            # ld_obj3 = load_object(file_path='artifacts/nlp_basic_obj_ess1.pkl')
            # ld_obj4 = load_object(file_path='artifacts/nlp_basic_obj_ess2.pkl')
            # ld_obj5 = load_object(file_path='artifacts/nlp_basic_obj_res_sum.pkl')
            # ld_obj6 = load_object(file_path='artifacts/gensim_w2v_title.pkl')
            # ld_obj7 = load_object(file_path='artifacts/gensim_w2v_ess1.pkl')
            # ld_obj8 = load_object(file_path='artifacts/gensim_w2v_ess2.pkl')
            # ld_obj9 = load_object(file_path='artifacts/gensim_w2v_res_sum.pkl')

            # print(ld_obj1.expensive_item_price_threshold)
            # print('-'*50)
            # print(ld_obj2.proj_title_rej_sub_pop_wrds.iloc[:10])
            # print('-'*50)
            # print(ld_obj3.proj_ess_rej_sub_pop_wrds.iloc[:10])
            # print('-'*50)
            # print(ld_obj4.proj_ess_rej_sub_pop_wrds.iloc[:10])
            # print('-'*50)
            # print(ld_obj5.proj_res_sum_rej_sub_pop_wrds.iloc[:10])
            # print('-'*50)
            # print(len(ld_obj6.model.wv.key_to_index))
            # print('-'*50)
            # print(len(ld_obj7.model.wv.key_to_index))
            # print('-'*50)
            # print(len(ld_obj8.model.wv.key_to_index))
            # print('-'*50)
            # print(len(ld_obj9.model.wv.key_to_index))

            

            # Save datasets post feature engineering:
            if save_data_post_feat_eng==1:

                print('Saving data post feature engineering start')
                print('-'*print_sep_len)

                # Creating a directory
                os.makedirs(os.path.dirname(self.train_non_nlp_path), exist_ok=True)

                train_non_nlp_df.to_csv(self.train_non_nlp_path, sep=',', index=False)
                test_non_nlp_df.to_csv(self.test_non_nlp_path, sep=',', index=False)

                train_nlp_basic_df.to_csv(self.train_nlp_basic_path, sep=',', index=False)
                test_nlp_basic_df.to_csv(self.test_nlp_basic_path, sep=',', index=False)

                train_nlp_w2v_df.to_csv(self.train_nlp_w2v_path, sep=',', index=False)
                test_nlp_w2v_df.to_csv(self.test_nlp_w2v_path, sep=',', index=False)

                train_final_df.to_csv(self.train_final_df_path, sep=',', index=False)
                test_final_df.to_csv(self.test_final_df_path, sep=',', index=False)

                print('Saving data post feature engineering end')
                print('-'*print_sep_len)


            if model_train==1:
                # Data Preprocessing, Model Training & Evaluation:
                modeltrainer_obj = ModelTrainer()
                modeltrainer_obj.train_model(train_df=train_final_df, test_df=test_final_df, sample_size=1, read_presaved_data=0)



        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  




if __name__=='__main__':

    print('Train Pipeline testing started')

    # train_pipe0 = TrainPipeline(sample_size=1, test_size=0.3, random_state=42)
    # # Save data post FE
    # train_pipe0.train_model_from_scratch(save_data_post_feat_eng=1, model_train=1)


    # Train model on presaved data (post FE)
    modeltrainer_obj = ModelTrainer()
    modeltrainer_obj.train_model(train_df=None, test_df=None, sample_size=1, read_presaved_data=1)


    # train_pipe1 = TrainPipeline(sample_n=2000, test_size=0.25, random_state=42)

    # # Dont save data post FE & dont train model: Just for checking pipeline
    # train_pipe1.train_model_from_scratch(save_data_post_feat_eng=0, model_train=0)

    # # Save data post FE and train model: Save data post FE on a sample from original data and train model
    # train_pipe1.train_model_from_scratch(save_data_post_feat_eng=1, model_train=1)





    


