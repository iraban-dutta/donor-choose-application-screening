import sys
import numpy as np
import pandas as pd
import time
import gensim
from gensim.utils import simple_preprocess
from scipy.stats import skew, kurtosis
from src.exception import CustomException
from src.components.data_preprocessing import DataPreProcessing



class FeatureEngineeringNLPW2V:

    def __init__(self):
        self.model = None
        

    def word2vec_gen_story(self, df, col):
        try: 
            story = []
            for lst in df[col].apply(simple_preprocess):
                story.append(lst)
            
            return story
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  


    def word2vec_train(self, story, col):
        try:
            start_time = time.time()
            self.model.build_vocab(story)
            self.model.train(story, total_examples=self.model.corpus_count, epochs=self.model.epochs)
            end_time = time.time()
            # print(f'Training time for feature {col}:', end_time-start_time)

        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  




    def train_gensim_model(self, df:pd.DataFrame, col:str, vec_size=100):
        '''
        This function creates and trains a gensim model on a single textual feature which has been cleaned
        
        Helper functions:
        - word2vec_gen_story
        - word2vec_train
        '''
        
        try:

            # Generate tokens per document
            story = self.word2vec_gen_story(df, col)
            
            # Generate gensim model
            if col=='project_title_cln':
                self.model = gensim.models.Word2Vec(window=5, min_count=2, workers=8, vector_size=vec_size)
            else:
                self.model = gensim.models.Word2Vec(window=10, min_count=2, workers=8, vector_size=vec_size)
                
            # Train gensim model
            self.word2vec_train(story, col)
            
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  



    def word2vec_gen_docvec(self, text):
        try: 
            wordvec_dim = self.model.wv[self.model.wv.index_to_key[0]].shape[0]
            # print(wordvec_dim, text)
            avg_vec = np.zeros(wordvec_dim)
            # This check is to account for words for whom vector is not created (min_count=2)
            doc = [wrd for wrd in simple_preprocess(text) if wrd in self.model.wv.key_to_index]
            # print(doc)
            if len(doc)>0:
                avg_vec = np.mean(self.model.wv[doc], axis=0)
            return avg_vec
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  




    def generate_doc_avg_w2v_arr(self, df:pd.DataFrame, col:str):
        
        '''
        Given a feature, this function creates an array where: 
        - each row represents a document and it is the average word2vec vector of all words present in the document (using the trained gensim model)
        
        Helper functions:
        - word2vec_gen_docvec
        '''
        
        try:
            arr = []
            start_time=time.time()
            for doc in df[col].values:
                arr.append(self.word2vec_gen_docvec(doc))
            end_time=time.time()
            print(f'Generating average word2vec for each document in {col}:', end_time-start_time)
                  
            return np.array(arr)
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  




    def generate_doc_avg_w2v_df(self, df:pd.DataFrame, col:str, prefix:str, sk_flag=0):
        
        '''
        Given a feature, this function creates a df where: 
        - each row represents a document and it is the average word2vec vector of all words present in the document (using the trained gensim model)
        - each row will also contain the skew and kurtosis of each average word2vec vector (if sk_flag is set to 1)
        
        Helper functions:
        - generate_doc_avg_w2v_arr
        - word2vec_gen_docvec
        '''
        
        try: 
            docvec_arr = self.generate_doc_avg_w2v_arr(df, col)
            docvec_df = pd.DataFrame(docvec_arr, index=df['id'])
            docvec_df.columns = [f'w2v_{prefix}_{col}' for col in docvec_df.columns]
            if sk_flag==1:
                docvec_df[f'{prefix}_skew'] = skew(docvec_arr, axis=1)
                docvec_df[f'{prefix}_kurt'] = kurtosis(docvec_arr, axis=1)
                
            return docvec_df
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  




if __name__=='__main__':

    print('Feature Engineering NLP-W2V testing started')

    # print_sep_len = 100

    # train_nlp_w2v_inp_df = pd.read_csv('artifacts/data_post_feat_eng/train_nlp_w2v_inp.csv')
    # test_nlp_w2v_inp_df = pd.read_csv('artifacts/data_post_feat_eng/test_nlp_w2v_inp.csv')
    # train_nlp_w2v_inp_df.fillna('', inplace=True)
    # test_nlp_w2v_inp_df.fillna('', inplace=True)

    # print(train_nlp_w2v_inp_df.shape, test_nlp_w2v_inp_df.shape)
    # print(train_nlp_w2v_inp_df.isna().sum())
    # print(test_nlp_w2v_inp_df.isna().sum())

    # fe_nlp_w2v_title_obj = FeatureEngineeringNLPW2V()
    # fe_nlp_w2v_title_obj.train_gensim_model(df=train_nlp_w2v_inp_df, col='project_title_cln', vec_size=100)
    # train_nlp_w2v_title_df = fe_nlp_w2v_title_obj.generate_doc_avg_w2v_df(df=train_nlp_w2v_inp_df, col='project_title_cln', prefix='title', sk_flag=1)
    # test_nlp_w2v_title_df = fe_nlp_w2v_title_obj.generate_doc_avg_w2v_df(df=test_nlp_w2v_inp_df, col='project_title_cln', prefix='title', sk_flag=1)
    
    # fe_nlp_w2v_ess1_obj = FeatureEngineeringNLPW2V()
    # fe_nlp_w2v_ess1_obj.train_gensim_model(df=train_nlp_w2v_inp_df, col='proj_essay1_cln', vec_size=100)
    # train_nlp_w2v_ess1_df = fe_nlp_w2v_ess1_obj.generate_doc_avg_w2v_df(df=train_nlp_w2v_inp_df, col='proj_essay1_cln', prefix='ess1', sk_flag=1)
    # test_nlp_w2v_ess1_df = fe_nlp_w2v_ess1_obj.generate_doc_avg_w2v_df(df=test_nlp_w2v_inp_df, col='proj_essay1_cln', prefix='ess1', sk_flag=1)

    # fe_nlp_w2v_ess2_obj = FeatureEngineeringNLPW2V()
    # fe_nlp_w2v_ess2_obj.train_gensim_model(df=train_nlp_w2v_inp_df, col='proj_essay2_cln', vec_size=100)
    # train_nlp_w2v_ess2_df = fe_nlp_w2v_ess2_obj.generate_doc_avg_w2v_df(df=train_nlp_w2v_inp_df, col='proj_essay2_cln', prefix='ess2', sk_flag=1)
    # test_nlp_w2v_ess2_df = fe_nlp_w2v_ess2_obj.generate_doc_avg_w2v_df(df=test_nlp_w2v_inp_df, col='proj_essay2_cln', prefix='ess2', sk_flag=1)

    # fe_nlp_w2v_res_sum_obj = FeatureEngineeringNLPW2V()
    # fe_nlp_w2v_res_sum_obj.train_gensim_model(df=train_nlp_w2v_inp_df, col='project_resource_summary_cln', vec_size=100)
    # train_nlp_w2v_res_sum_df = fe_nlp_w2v_res_sum_obj.generate_doc_avg_w2v_df(df=train_nlp_w2v_inp_df, col='project_resource_summary_cln', prefix='res_sum', sk_flag=1)
    # test_nlp_w2v_res_sum_df = fe_nlp_w2v_res_sum_obj.generate_doc_avg_w2v_df(df=test_nlp_w2v_inp_df, col='project_resource_summary_cln', prefix='res_sum', sk_flag=1)


    # # print(train_nlp_w2v_title_df.shape, test_nlp_w2v_title_df.shape)
    # # print(train_nlp_w2v_ess1_df.shape, test_nlp_w2v_ess1_df.shape)
    # # print(train_nlp_w2v_ess2_df.shape, test_nlp_w2v_ess2_df.shape)
    # # print(train_nlp_w2v_res_sum_df.shape, test_nlp_w2v_res_sum_df.shape)


    # data_preprocess_obj = DataPreProcessing()
    # train_nlp_w2v_df = data_preprocess_obj.merge_nlp_w2v_feats(df_title=train_nlp_w2v_title_df, df_ess1=train_nlp_w2v_ess1_df, 
    #                                                            df_ess2=train_nlp_w2v_ess2_df, df_res_sum=train_nlp_w2v_res_sum_df)
    # test_nlp_w2v_df = data_preprocess_obj.merge_nlp_w2v_feats(df_title=test_nlp_w2v_title_df, df_ess1=test_nlp_w2v_ess1_df, 
    #                                                           df_ess2=test_nlp_w2v_ess2_df, df_res_sum=test_nlp_w2v_res_sum_df)
    # train_nlp_w2v_df.fillna(0, inplace=True)
    # test_nlp_w2v_df.fillna(0, inplace=True)


    # print(train_nlp_w2v_df.shape, test_nlp_w2v_df.shape)
    # print(train_nlp_w2v_df.isna().sum().loc[train_nlp_w2v_df.isna().sum()>0])
    # print(test_nlp_w2v_df.isna().sum().loc[test_nlp_w2v_df.isna().sum()>0])
    # print(train_nlp_w2v_df.head(2))
    # print(test_nlp_w2v_df.head(2))




