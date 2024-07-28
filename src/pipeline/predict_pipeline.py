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
from src.utils import get_cross_val_score_summary, get_classification_report, save_object, load_object






class UserInputCompileProjectData:

    def __init__(self,
                 teacher_prefix: str,
                 school_state: str,
                 project_grade_category: str,
                 project_subject_categories: str,
                 project_title: str,
                 project_essay_1: str,
                 project_essay_2: str,
                 project_resource_summary: str):

        self.teacher_prefix = teacher_prefix
        self.school_state = school_state
        self.project_grade_category = project_grade_category
        self.project_subject_categories = project_subject_categories
        self.project_title = project_title
        self.project_essay_1 = project_essay_1
        self.project_essay_2 = project_essay_2
        self.project_resource_summary = project_resource_summary


    def prepare_df_for_predict(self):

        try:

            df_dict = {

                # # Hard-coded Features: Debug Mode
                # 'Unnamed: 0' : 103020,
                # 'id': 'p092944',
                # 'teacher_id': '87bc1584ca9d5bfe79365c5427a0bc14',
                # 'teacher_number_of_previously_posted_projects': 1,
                # 'project_submitted_datetime' : '2017-04-12 12:01:51',
                # 'project_subject_subcategories': 'Team Sports',
                # 'project_essay_3': '',
                # 'project_essay_4': '',
                # 'project_is_approved': 1, # This does not affect prediction, using just for consistency of df.shape


                # Hard-coded Features
                'Unnamed: 0' : 51266,
                'id': 'p218094',
                'teacher_id': '00000f7264c27ba6fea0c837ed6aa0aa',
                'teacher_number_of_previously_posted_projects': 2,
                'project_submitted_datetime' : '2017-01-09 19:01:51',
                'project_subject_subcategories': 'Literacy',
                'project_essay_3': '',
                'project_essay_4': '',
                'project_is_approved': 1, # This does not affect prediction, using just for consistency of df.shape


                # User Inputs
                'teacher_prefix' : self.teacher_prefix,
                'school_state' : self.school_state,
                'project_grade_category' : self.project_grade_category,
                'project_subject_categories' : self.project_subject_categories,
                'project_title' : self.project_title,
                'project_essay_1' : self.project_essay_1,
                'project_essay_2' : self.project_essay_2,
                'project_resource_summary' : self.project_resource_summary

            }

            return pd.DataFrame([df_dict])
        

        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)

    

class UserInputResourcetData:

    def __init__(self, res_description: str, res_quantity: int, res_price: float):

        self.id = 'p218094'
        self.res_description = res_description
        self.res_quantity = int(res_quantity)
        self.res_price = float(res_price)

    def to_dict(self):

        return {'id':self.id, 'description': self.res_description, 'quantity': self.res_quantity, 'price': self.res_price}
        


class UserInputCompileResourcetData:

    def __init__(self):
        # List of dictionaries --> Each dictionary will have the details of each resource
        self.res_list = []

    def append_resource(self, resource_dict: dict):
        try:
            self.res_list.append(resource_dict)

        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)


    def prepare_df_for_predict(self):

        try:

            return pd.DataFrame(self.res_list)
    
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)




class PredictOnUserInput:

    def __init__(self):
        pass


    def predict(self, predict_df: pd.DataFrame, predict_df_res: pd.DataFrame):

        try: 

            print_sep_len = 100



            # Perform Data Cleaning
            data_clean_obj = DataCleaning()
            predict_non_text_df = data_clean_obj.clean_nontext_feat(df_inp=predict_df)
            predict_text_df = data_clean_obj.clean_text_feat(df_inp=predict_df)

            data_clean_obj.fit_flag=1
            predict_res_df = data_clean_obj.clean_res(df_inp=predict_df_res, fit=0)

            print(predict_non_text_df.shape)
            print(predict_non_text_df.columns)
            print('-'*print_sep_len)
            print(predict_text_df.shape)
            print(predict_text_df.columns)
            print('-'*print_sep_len)
            print(predict_res_df.shape)
            print(predict_res_df.columns)
            print('-'*print_sep_len)



            # Loading fitted objects for feature engineering
            fitted_obj_non_nlp = load_object(file_path='artifacts/non_nlp_obj.pkl')

            fitted_obj_nlp_basic_title = load_object(file_path='artifacts/nlp_basic_obj_title.pkl')
            fitted_obj_nlp_basic_ess1 = load_object(file_path='artifacts/nlp_basic_obj_ess1.pkl')
            fitted_obj_nlp_basic_ess2 = load_object(file_path='artifacts/nlp_basic_obj_ess2.pkl')
            fitted_obj_nlp_basic_res_sum = load_object(file_path='artifacts/nlp_basic_obj_res_sum.pkl')

            fitted_obj_nlp_w2v_title = load_object(file_path='artifacts/gensim_w2v_title.pkl')
            fitted_obj_nlp_w2v_ess1 = load_object(file_path='artifacts/gensim_w2v_ess1.pkl')
            fitted_obj_nlp_w2v_ess2 = load_object(file_path='artifacts/gensim_w2v_ess2.pkl')
            fitted_obj_nlp_w2v_res_sum = load_object(file_path='artifacts/gensim_w2v_res_sum.pkl')

            print('Fitted objects for Feature Engineering Loaded')
            print('-'*print_sep_len)


            # Generate Features : Non NLP
            predict_non_nlp_df = fitted_obj_non_nlp.gen_non_nlp_feats(df_inp1=predict_non_text_df, df_inp2=predict_res_df, fit=0, test_mode=0)


            # Generate Features : NLP Basic
            predict_title_df = fitted_obj_nlp_basic_title.gen_nlp_basic_title_feats(df_inp=predict_text_df, fit=0, test_mode=0)
            predict_ess1_df = fitted_obj_nlp_basic_ess1.gen_nlp_basic_ess_feats(df_inp=predict_text_df, col='proj_essay1', fit=0, test_mode=0)
            predict_ess2_df = fitted_obj_nlp_basic_ess2.gen_nlp_basic_ess_feats(df_inp=predict_text_df, col='proj_essay2', fit=0, test_mode=0)
            fe_nlp_basic_ess_sim_obj = FeatureEngineeringNLPBasic2()
            predict_ess_sim_df = fe_nlp_basic_ess_sim_obj.gen_nlp_basic_ess_similar_feats(df_ess1=predict_ess1_df, df_ess2=predict_ess2_df, test_mode=0)
            predict_res_sum_df = fitted_obj_nlp_basic_res_sum.gen_nlp_basic_res_sum_feats(df_inp=predict_text_df, fit=0, test_mode=0)

            
            data_preprocess_obj1 = DataPreProcessing()
            predict_merge_nlp_basic_df = data_preprocess_obj1.merge_nlp_basic_feats(df_title=predict_title_df, 
                                                                                    df_ess1=predict_ess1_df, 
                                                                                    df_ess2=predict_ess2_df, 
                                                                                    df_ess_sim=predict_ess_sim_df, 
                                                                                    df_res_sum=predict_res_sum_df)
            predict_nlp_basic_df = data_preprocess_obj1.prep_nlp_basic_df(df_inp=predict_merge_nlp_basic_df)


            # Generate Features : NLP-W2V
            data_preprocess_obj2 = DataPreProcessing()
            predict_nlp_w2v_inp_df = data_preprocess_obj2.prep_nlp_w2v_df(df_inp=predict_merge_nlp_basic_df)

            # Some samples in the textual features (post cleaning) can become missing (due to stopword removal)
            predict_nlp_w2v_inp_df.fillna('', inplace=True)

            predict_nlp_w2v_title_df = fitted_obj_nlp_w2v_title.generate_doc_avg_w2v_df(df=predict_nlp_w2v_inp_df, col='project_title_cln', prefix='title', sk_flag=1)
            predict_nlp_w2v_ess1_df = fitted_obj_nlp_w2v_ess1.generate_doc_avg_w2v_df(df=predict_nlp_w2v_inp_df, col='proj_essay1_cln', prefix='ess1', sk_flag=1)
            predict_nlp_w2v_ess2_df = fitted_obj_nlp_w2v_ess2.generate_doc_avg_w2v_df(df=predict_nlp_w2v_inp_df, col='proj_essay2_cln', prefix='ess2', sk_flag=1)
            predict_nlp_w2v_res_sum_df = fitted_obj_nlp_w2v_res_sum.generate_doc_avg_w2v_df(df=predict_nlp_w2v_inp_df, col='project_resource_summary_cln', prefix='res_sum', sk_flag=1)
            predict_nlp_w2v_df = data_preprocess_obj2.merge_nlp_w2v_feats(df_title=predict_nlp_w2v_title_df, df_ess1=predict_nlp_w2v_ess1_df, 
                                                                          df_ess2=predict_nlp_w2v_ess2_df, df_res_sum=predict_nlp_w2v_res_sum_df)
            predict_nlp_w2v_df.fillna(0, inplace=True)


            # Stacking all features together
            data_preprocess_obj3 = DataPreProcessing()
            predict_final_df = data_preprocess_obj3.merge_all_feats(df_non_nlp=predict_non_nlp_df, df_nlp_basic=predict_nlp_basic_df, df_nlp_w2v=predict_nlp_w2v_df)
            predict_final_df_post_fe = predict_final_df.copy()

            # Dropping irrelevant columns
            predict_final_df = predict_final_df.drop(['id', 'project_is_approved'], axis=1).copy()

            # Sanity checks on data
            print(predict_non_nlp_df.shape)
            print(predict_non_nlp_df.columns)
            print('-'*print_sep_len)
            print(predict_nlp_basic_df.shape)
            print(predict_nlp_basic_df.columns)
            print('-'*print_sep_len)
            print(predict_nlp_w2v_df.shape)
            print(predict_nlp_w2v_df.columns)
            print('-'*print_sep_len)
            print(predict_final_df.shape)
            print(predict_final_df.columns)
            print('-'*print_sep_len)


            print('Missing values:')
            print(predict_non_nlp_df.isna().sum().loc[predict_non_nlp_df.isna().sum()>0])
            print(predict_nlp_basic_df.isna().sum().loc[predict_nlp_basic_df.isna().sum()>0])
            print(predict_nlp_w2v_df.isna().sum().loc[predict_nlp_w2v_df.isna().sum()>0])
            print(predict_final_df.isna().sum().loc[predict_final_df.isna().sum()>0])
            print('-'*print_sep_len)


            # Loding fitted objects for Cat2Num 
            fitted_enc_target_teac_prefix = load_object(file_path='artifacts/enc_target_teacher_prefix.pkl')
            fitted_enc_target_school_state = load_object(file_path='artifacts/enc_target_school_state.pkl')
            fitted_enc_res_exp_item_qcat = load_object(file_path='artifacts/enc_target_res_exp_item_qcat.pkl')
            fitted_enc_project_grade_cat = load_object(file_path='artifacts/enc_target_project_grade_category.pkl')
            fitted_enc_records_per_user_cat = load_object(file_path='artifacts/enc_label_records_per_user_cat.pkl')

            # Performing Cat2Num Transformation
            predict_final_df['teacher_prefix'] = fitted_enc_target_teac_prefix.transform(predict_final_df['teacher_prefix'])
            predict_final_df['school_state'] = fitted_enc_target_school_state.transform(predict_final_df['school_state'])
            predict_final_df['res_exp_item_qcat'] = fitted_enc_res_exp_item_qcat.transform(predict_final_df['res_exp_item_qcat'])
            predict_final_df['project_grade_category'] = fitted_enc_project_grade_cat.transform(predict_final_df['project_grade_category'])
            predict_final_df['records_per_user_cat'] = fitted_enc_records_per_user_cat.transform(predict_final_df['records_per_user_cat'])


            print('Cat2Num Transformation done')
            print(predict_final_df.shape)
            print(predict_final_df.columns)
            print('-'*print_sep_len)
            count = 0
            for col in predict_final_df.columns:
                if predict_final_df[col].dtype=='object':
                    print(col)
                    count += 1
            print('Total categorical columns found:', count)
            print('-'*print_sep_len)


            # Loding fitted object for Feature scaling
            fitted_feat_scaler = load_object(file_path='artifacts/scaler_features.pkl')
            # print(fitted_feat_scaler.feature_names_in_)

            # Ensuring the feature names are in the same order as they were when the standard scaler object was fitted
            predict_final_df_mapped = predict_final_df[fitted_feat_scaler.feature_names_in_]

            # Performing Feature Scaling
            predict_final_scl = fitted_feat_scaler.transform(predict_final_df_mapped)


            print('Feature scaling done')
            print(predict_final_scl.shape)
            print('-'*print_sep_len)


            # # DEBUG:
            # # Post FE:
            # predict_final_df_post_fe.to_csv('artifacts/predict_fe.csv', sep=',', index=False)
            # # Post Cat2Num
            # predict_final_df_mapped.to_csv('artifacts/predict_cat2num.csv', sep=',', index=False)
            # # Post Feature Scaling
            # pd.DataFrame(predict_final_scl).to_csv('artifacts/predict_scl.csv', sep=',', index=False)


            # Loading trained model for prediction
            trained_model = load_object(file_path='artifacts/model.pkl')
            print('Model Loaded')
            print('-'*print_sep_len)

            # Prediction
            prediction = trained_model.predict(predict_final_scl)[0]
            prediction_proba = np.round(trained_model.predict_proba(predict_final_scl)[0][1], 2)
            print('Prediction Completed')
            print('-'*print_sep_len)

            print('Prediction:', prediction, prediction_proba)

            if prediction_proba>=0.8:
                return f'Congrats! Project is likely to be approved with probability {str(prediction_proba)}', 'congrats'
            elif prediction_proba>=0.5:
                return f'Project is likely to be approved with probability {str(prediction_proba)}. \n However you can try with refining your proposal!', 'refine'
            elif prediction_proba<0.5:
                return f'Sorry! Project is NOT likely to be approved, approval probability is {str(prediction_proba)}. \n Please refine your proposal!', 'sorry'
            else:
                return f'Application Down!', 'error'



        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)


 



if __name__=='__main__':
    pass




