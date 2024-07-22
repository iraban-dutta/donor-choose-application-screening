import sys
import numpy as np
import pandas as pd
import re
import string
import emoji
from collections import Counter
from nltk.stem.porter import PorterStemmer
from src.constants import STOPWORDS_ENGLISH
from src.exception import CustomException
from src.components.data_ingestion import DataIngest
from src.components.data_cleaning import DataCleaning

from tqdm import tqdm
tqdm.pandas()




class FeatureEngineeringNLPBasic1:

    def __init__(self):
        self.proj_title_rej_sub_pop_wrds = pd.Series(dtype='float')


    def exclusive_pop_words_rej(self, df, col):
    
        try: 
            # Rejected submissions: Generate corpus
            rej_corpus = []
            for lst in df.loc[df['project_is_approved']==0, col].str.split():
                rej_corpus += lst
            
            # Rejected submissions: Generate probability for each word in vocabulary
            rej_vocab = Counter(rej_corpus)
            rej_vocab = pd.Series(rej_vocab).sort_values(ascending=False)
            rej_vocab_prob = rej_vocab/rej_vocab.sum()
            
            
            # Accepted submissions: Generate corpus
            acc_corpus = []
            for lst in df.loc[df['project_is_approved']==1, col].str.split():
                acc_corpus += lst
            
            # Accepted submissions: Generate probability for each word in vocabulary
            acc_vocab = Counter(acc_corpus)
            acc_vocab = pd.Series(acc_vocab).sort_values(ascending=False)
            acc_vocab_prob = acc_vocab/acc_vocab.sum()
            
            # We want to generate a score against each word such that:
                # ser1 = rej_vocab_prob, ser2 = acc_vocab_prob
                # If word is explicitly present in ser1, score will be high and positive
                # If word is common to both ser1 & ser2, score will be close to zero
                # If word is explicitly present in ser2, score will be high and negative
            
            
            dict1 = dict(zip(rej_vocab_prob.index, rej_vocab_prob.values))
            dict2 = dict(zip(acc_vocab_prob.index, acc_vocab_prob.values))
            
            tokens = {}
            
            for key1 in dict1:
                if key1 in dict2:
                    tokens[key1] = round(dict1[key1] - dict2[key1], 6) # tokens common to both series: ser1 & ser2
                else:
                    tokens[key1] = dict1[key1] # tokens explicitly in ser1: prob of ser1(word)
            
            for key2 in dict2:
                if key2 not in dict1:
                    tokens[key2] = -dict2[key2]  # tokens explicitly in ser2: prob of ser2(word)

            tokens = pd.Series(tokens)
            tokens = pd.Series(tokens).sort_values(ascending=False)
            tokens_prob = tokens/tokens.sum()
            
            return tokens_prob  

        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)        



    def adv_feat_rej_scr(self, row, col, rej_corpus):
        
        try: 
            # Split into tokens
            text = row[col]
            tokens = text.split()
            
            if len(tokens)==0:
                return 0
            
            # Number of exclusive & popular words from rejected submissions
            count = sum([1 for tkn in tokens if tkn in rej_corpus])
            
            count /= len(tokens)
            
            return count 
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  




    

    def naive_features_title_token(self, text):

        try: 
            token_features = [0.0]*13
            
            if len(text)==0:
                return token_features
            
            # Handle escape sequences
            text = re.sub(r'\\[nrt]', ' ', text.strip())
            
            # Count emojis
            emojis = sum([emoji.is_emoji(char) for char in text])
            text1 = ''.join([char for char in text if emoji.is_emoji(char) is False])
            
            # Count Emoticons
            emoticon_pattern = r'[:;=8][-~]?[)D(|/\\OpP3]'
            emoticons = len(re.findall(emoticon_pattern, text1))
            text2 = re.sub(emoticon_pattern, '', text1)
            
            # Count Punctuations
            punctuation_set = set(string.punctuation)
            punctuations =  len([char for char in text2 if char in punctuation_set])
            text3 = ''.join([char for char in text2 if char not in punctuation_set])

            # Count completely uppercase words
            capwords = sum([word.isupper() for word in text3.split()])
            
            
            # Lowercasing and removing extra white spaces
            text4 = (' ').join([word.strip() for word in text3.lower().split()])
            tokens = text4.split()
            
            
            # # Identify Abbreviations or Colloquialisms (Taking a lot of time)
            # english_words_set = set(words.words())
            # abbreviations = [word for word in tokens if word.strip() not in english_words_set]
            
            # Stop words in English
            stop_words_eng = STOPWORDS_ENGLISH
            
            # Stop and non stop words in text
            stop_words = [word for word in tokens if word in stop_words_eng]
            non_stop_words = [word for word in tokens if word not in stop_words_eng]
            
            
            token_features[0] = emojis # Number of emojis
            token_features[1] = emoticons # Number of emoticons
            token_features[2] = punctuations # Number of punctuations
            token_features[3] = capwords # Number of capwords
            token_features[4] = round(capwords/len(tokens), 3) # Ratio capwords/words
            token_features[5] = len(tokens) # Number of words
            token_features[6] = len(text4) # Number of characters
            token_features[7] = round(len(text4)/len(tokens), 3) # Ratio characters/words
            # token_features[8] = len(abbreviations) # Number of abbreviations
            token_features[9] = len(stop_words) # Number of stop words
            token_features[10] = len(non_stop_words) # Number of non stop words
            token_features[11] = round(len(stop_words)/len(tokens), 3) # Ratio stop words/words
            token_features[12] = text4 # Primary pre-processed text
            

            return token_features
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  



    def preprocess_remove_stop_stem_title(self, text):

        try: 
        
            # Stop words
            stop_words_eng = STOPWORDS_ENGLISH
            
            # Remove stop words
            text1 = (' ').join([word if word not in stop_words_eng else '' for word in text.split()])
            
            
            # Stemming
            ps = PorterStemmer()
            text2 = (' ').join([ps.stem(word.strip()) for word in text1.split()])
            
            return text2
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  







    def gen_nlp_basic_title_feats(self, df_inp:pd.DataFrame, fit=0, test_mode=0):
    
        '''
        This function extracts the below nlp features from the textual feature: 'project_title'
        - 'title_emoj'         : Number of emojis
        - 'title_emot'         : Number of emoticons
        - 'title_punc'         : Number of punctuations
        - 'title_capwords'     : Number of capwords
        - 'title_capwords_r'   : Ratio capwords/words
        - 'title_words'        : Number of words
        - 'title_chars'        : Number of characters
        - 'title_chars_r'      : Ratio characters/words
        - 'title_stopwords'    : Number of stop words
        - 'title_nonstopwords' : Number of non stop words
        - 'title_stopwords_r'  : Ratio stop words/words
        - 'title_rej_scr'      : Score capturing presence of popular words found in this feature of rejected submissions
        - 'project_title_cln'  : Clean feature after perfroming basic preprocessing (includes stop word removal + stemming)
        
        Inputs: 
        - Main dataframe
        - fit: 0/1
        
        Helper Methods used:
        - naive_features_title_token
        - preprocess_remove_stop_stem_title
        - exclusive_pop_words_rej
        - adv_feat_rej_scr

        Actions:
        - fit=1 : To be used with train data, fits the object first by learning parameters from the train data and uses it to perform feature engineering on the train data
        - fit=0 : To be used with test data, using the learnt parameters from train data, performs feature engineering on the test data
        
        '''
        
        try: 
            if test_mode==1:
                df_inp = df_inp.iloc[:1000].copy()

            print('Generating NLP-Basic features: project_title')
            df_text1 = df_inp[['id', 'project_title', 'project_is_approved']].copy()
            
            # Perform primary preprocessing & generate naive token features
            proj_title_naive_token_features = df_text1['project_title'].progress_apply(self.naive_features_title_token)
            print('Primary preprocessing & Generating Naive Token Features Done!')
            
            df_text1['title_emoj'] = list(map(lambda x: x[0], proj_title_naive_token_features))
            df_text1['title_emot'] = list(map(lambda x: x[1], proj_title_naive_token_features))
            df_text1['title_punc'] = list(map(lambda x: x[2], proj_title_naive_token_features))
            df_text1['title_capwords'] = list(map(lambda x: x[3], proj_title_naive_token_features))
            df_text1['title_capwords_r'] = list(map(lambda x: x[4], proj_title_naive_token_features))
            df_text1['title_words'] = list(map(lambda x: x[5], proj_title_naive_token_features))
            df_text1['title_chars'] = list(map(lambda x: x[6], proj_title_naive_token_features))
            df_text1['title_chars_r'] = list(map(lambda x: x[7], proj_title_naive_token_features))
            df_text1['title_stopwords'] = list(map(lambda x: x[9], proj_title_naive_token_features))
            df_text1['title_nonstopwords'] = list(map(lambda x: x[10], proj_title_naive_token_features))
            df_text1['title_stopwords_r'] = list(map(lambda x: x[11], proj_title_naive_token_features))
            df_text1['project_title_pp1'] = list(map(lambda x: x[12], proj_title_naive_token_features))
            
            # Perform advanced preprocessing: Stop word removal + stemming
            df_text1['project_title_cln'] = df_text1['project_title_pp1'].progress_apply(self.preprocess_remove_stop_stem_title)
            print('Advanced preprocessing Done!')
            
            # Extract Feature: Title Rejection Score
            if fit==1:
                # Step1: Extract popular words in project title in rejected submissions
                self.proj_title_rej_sub_pop_wrds = self.exclusive_pop_words_rej(df_text1, 'project_title_cln')

            if len(self.proj_title_rej_sub_pop_wrds)==0:
                raise Exception('Object not fitted yet!')
            # Step2: Creating feature (Limiting rejected vocabulary to the top 3000 most popular words present in rejected submissions)
            df_text1['title_rej_scr'] = df_text1.progress_apply(lambda x: self.adv_feat_rej_scr(x, 'project_title_cln', self.proj_title_rej_sub_pop_wrds.iloc[:3000].index), axis=1)
            print('Generating Rejection Score Done!')
            
            # Dropping unnecessary columns
            df_text1.drop(['project_title', 'project_title_pp1', 'project_is_approved'], axis=1, inplace=True)
            
            return df_text1
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)






if __name__=='__main__':

    print('Feature Engineering NLP-Basic started: Project Title!')

    # print(STOPWORDS_ENGLISH[:100])
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



    # fe_nlp_basic_title_obj = FeatureEngineeringNLPBasic1()

    # train_title_df = fe_nlp_basic_title_obj.gen_nlp_basic_title_feats(df_inp=train_text_df, fit=1, test_mode=0)
    # test_title_df = fe_nlp_basic_title_obj.gen_nlp_basic_title_feats(df_inp=test_text_df, fit=0, test_mode=0)
    # print('-'*print_sep_len)
    # print('NLP-Basic DF:')
    # print(train_title_df.shape)
    # print(train_title_df.columns)
    # print(train_title_df.isna().sum())
    # print('-'*print_sep_len)
    # print(test_title_df.shape)
    # print(test_title_df.columns)
    # print(test_title_df.isna().sum())
    


    