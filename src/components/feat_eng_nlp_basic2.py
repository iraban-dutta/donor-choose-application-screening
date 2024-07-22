import sys
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from src.constants import STOPWORDS_ENGLISH, CONTRACTIONS_ENGLISH
import distance
from fuzzywuzzy import fuzz
from src.exception import CustomException
from src.components.data_ingestion import DataIngest
from src.components.data_cleaning import DataCleaning

from tqdm import tqdm
tqdm.pandas()




class FeatureEngineeringNLPBasic2:

    def __init__(self):
        self.proj_ess_rej_sub_pop_wrds = pd.Series(dtype='float')


    def preprocess1(self, text):

        try: 
            # Replace certain special characters with their string equivalents
            text1 = text.strip()
            text1 = text1.replace('%', ' percent')
            text1 = text1.replace('$', ' dollar ')
            text1 = text1.replace('₹', ' rupee ')
            text1 = text1.replace('€', ' euro ')
            text1 = text1.replace('@', ' at ')
            text1 = text1.replace('title i', 'title 1')
            
            # Handle escape sequences
            text2 = re.sub(r'\\[nrt]', ' ', text1)
            # print(text2)
            
            # HTML tags & Obvious pattern removals (eg. url, emailID) --> Not required here
            # bs = BeautifulSoup(text2)
            # text3 = bs.get_text()
            
            # Sentence tokenizer
            sentences = sent_tokenize(text2)
            # print(sentences)
            
            processed_sentences = []
            for sentence in sentences:
                # Remove all punctuation except for sentence delimiters
                sentence = re.sub(r'[^\w\s\.\?\!]', '', sentence)
                # print(sentence)
                # Converting uppercase letter at sentence start to lowercase
                if len(sentence)>0:
                    sentence_copy = sentence[:]
                    sentence = sentence_copy[0].lower() + sentence_copy[1:]
                processed_sentences.append(sentence)
            
            text4 = (' ').join(processed_sentences)
            
            return text4
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  


    def naive_features_capwords(self, row, col):

        try:
            capwords_features = [0.0]*2
            text = row[col]
            
            # Counting total words
            total_words = len(text.split())
            
            # Ignore words of length=1
            trunc_word_lst = list(filter(lambda x: True if len(x)>1 else False, text.split()))
            
            # Count words that are capitalized
            cap_words = sum([1 if word[0].isupper() else 0 for word in trunc_word_lst])
            
            capwords_features[0] = cap_words # Number of cap words
            capwords_features[1] = round(cap_words/total_words, 3) # Ratio: cap words/total words
            
            return capwords_features
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)          


    def preprocess2(self, text):
        try:
        
            # Dictionary of contractions
            contractions = CONTRACTIONS_ENGLISH
            
            # Lowercasing and basic preprocess
            text1 = text.lower().strip()
            
            # Data specific fixes
            text1 = text1.replace('title i', 'title 1')

            # Decontracting words and removing extra white space  
            text2 = []
            for word in text1.split(): 
                word=word.strip()
                if word in contractions:
                    decontracted_word = contractions[word]
                else:
                    decontracted_word = word
                text2.append(decontracted_word)
            text2 = (' ').join(text2)
            
            return text2
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  


    def naive_features_token(self, row, col):
        
        try: 
            # Stop words
            stop_words_eng = STOPWORDS_ENGLISH
            
            token_features = [0.0]*10
            
            # Retrieve essay1 into a variable called text
            text = row[col]
            
            # Generate sentence list
            sentences = sent_tokenize(text)
            # print(sentences)
            
            # Generate word list
            words = []
            for sent in sentences:
                words.extend(sent.split())
            # print(words)
            
            # Estimate the count of question marks and exclamations used
            question_counter = 0
            exclamation_counter = 0
            question_counter = np.array(['?' in wrd for wrd in words]).sum()
            exclamation_counter = np.array(['!' in wrd for wrd in words]).sum()
            # print(question_counter, exclamation_counter)
            
            # Get rid of specific delimitters in words:' .?!'
            words = [wrd.strip(' .?!') for wrd in words]
            # print(words)
            
            # Stopword count
            stop_words = [word for word in words if word in stop_words_eng]
            
            # Non-Stop count
            nonstop_words = [word for word in words if word not in stop_words_eng]
            
            token_features[0] = len(sentences) # Number of sentences
            token_features[1] = len(words) # Number of words
            token_features[2] = len(text) # Number of characters
            token_features[3] = round(len(words)/len(sentences), 3) # Ratio: words/sentences
            token_features[4] = round(len(text)/len(words), 3) # Ratio: characters/words
            token_features[5] = len(stop_words) # Number of stop words
            token_features[6] = len(nonstop_words) # Number of non stop words
            token_features[7] = round(len(stop_words)/len(words), 3) # Ratio: stop words/words
            token_features[8] = question_counter # Number of question marks
            token_features[9] = exclamation_counter # Number of exclamation marks

            return token_features
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  


    def preprocess_remove_stop_stem(self, text):
        
        try:
            # Stop words
            stop_words_eng = STOPWORDS_ENGLISH
            
            # Removing all punctuations
            punctuation_set = set(string.punctuation)
            text1 = ''.join([char for char in text if char not in punctuation_set])
            
            # Remove stop words
            text2 = (' ').join([word if word not in stop_words_eng else '' for word in text1.split()])
            
            # Stemming
            ps = PorterStemmer()
            text3 = (' ').join([ps.stem(word.strip()) for word in text2.split()])
            
            return text3
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  


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




    def gen_nlp_basic_ess_feats(self, df_inp:pd.DataFrame, col='proj_essay1', fit=0, test_mode=0):
        
        '''
        This function extracts the below nlp features from the textual feature: 'project_essay'
        - 'ess_capwords'     : Number of capwords
        - 'ess_capwords_r'   : Ratio cap words/total words
        - 'ess_sents'        : Number of sentences
        - 'ess_words'        : Number of words
        - 'ess_chars'        : Number of characters
        - 'ess_word_sent_r'  : Ratio words/sentences
        - 'ess_char_word_r'  : Ratio characters/words
        - 'ess_stopwords'    : Number of stop words
        - 'ess_nonstopwords' : Number of non stop words
        - 'ess_stopwords_r'  : Ratio stop words/words
        - 'ess_ques_mark'    : Number of question marks
        - 'ess_excl_mark'    : Number of exclamation marks
        - 'ess_rej_scr'      : Score capturing presence of popular words found in this feature of rejected submissions
        - 'proj_essay_cln'   : Clean feature after perfroming basic preprocessing (includes stop word removal + stemming)
        
        Inputs: 
        - Main dataframe
        - col name: 'proj_essay1' or 'proj_essay2'
        - fit: 0/1
        
        Helper Functions used:
        - preprocess1
        - naive_features_capwords
        - preprocess2
        - naive_features_token
        - preprocess_remove_stop_stem
        - exclusive_pop_words_rej
        - adv_feat_rej_scr

        Actions:
        - fit=1 : To be used with train data, fits the object first by learning parameters from the train data and uses it to perform feature engineering on the train data
        - fit=0 : To be used with test data, using the learnt parameters from train data, performs feature engineering on the test data
        
        '''
        

        try: 
            if col=='proj_essay1':
                idx = 1
                subset_col = ['id', 'proj_essay1', 'project_is_approved']
            elif col=='proj_essay2':
                idx = 2 
                subset_col = ['id', 'proj_essay2', 'project_is_approved']
            else:
                print('Incorrect column given in parameters')
                return
            
            
            if test_mode==1:
                df_inp = df_inp.iloc[:1000].copy()
                
            
            df_text2 = df_inp[subset_col].copy()
            
            print(f'Generating NLP-Basic features: project_essay{idx}')
            # Perform primary preprocessing: Part I
            df_text2['proj_essay_pp1'] = df_text2[col].progress_apply(self.preprocess1)
            print('Primary preprocessing (Part I) Done!')
            
            # Creating the capword features
            proj_essay_capwords_features = df_text2.progress_apply(lambda x: self.naive_features_capwords(x, 'proj_essay_pp1'), axis=1)
            df_text2[f'ess{idx}_capwords'] = list(map(lambda x: x[0], proj_essay_capwords_features))
            df_text2[f'ess{idx}_capwords_r'] = list(map(lambda x: x[1], proj_essay_capwords_features))
            print('Generating Capwords Features Done!')
            
            # Perform primary preprocessing: Part II
            df_text2['proj_essay_pp2'] = df_text2['proj_essay_pp1'].progress_apply(self.preprocess2)
            print('Primary preprocessing (Part II) Done!')
            
            # Generating naive token features for essay
            proj_essay_naive_token_features = df_text2.progress_apply(lambda x: self.naive_features_token(x, 'proj_essay_pp2'), axis=1)
            print('Generating Naive Token Features Done!')
            
            df_text2[f'ess{idx}_sents'] = list(map(lambda x: x[0], proj_essay_naive_token_features))
            df_text2[f'ess{idx}_words'] = list(map(lambda x: x[1], proj_essay_naive_token_features))
            df_text2[f'ess{idx}_chars'] = list(map(lambda x: x[2], proj_essay_naive_token_features))
            df_text2[f'ess{idx}_word_sent_r'] = list(map(lambda x: x[3], proj_essay_naive_token_features))
            df_text2[f'ess{idx}_char_word_r'] = list(map(lambda x: x[4], proj_essay_naive_token_features))
            df_text2[f'ess{idx}_stopwords'] = list(map(lambda x: x[5], proj_essay_naive_token_features))
            df_text2[f'ess{idx}_nonstopwords'] = list(map(lambda x: x[6], proj_essay_naive_token_features))
            df_text2[f'ess{idx}_stopwords_r'] = list(map(lambda x: x[7], proj_essay_naive_token_features))
            df_text2[f'ess{idx}_ques_mark'] = list(map(lambda x: x[8], proj_essay_naive_token_features))
            df_text2[f'ess{idx}_excl_mark'] = list(map(lambda x: x[9], proj_essay_naive_token_features))

            # Perform advanced preprocessing: Stop word removal + stemming
            df_text2[f'proj_essay{idx}_cln'] = df_text2['proj_essay_pp2'].progress_apply(self.preprocess_remove_stop_stem)
            print('Advanced preprocessing Done!')
            
            
            # Extract Feature: Essay Rejection Score
            
            if fit==1:
                # Step1: Extract popular words in project essay in rejected submissions
                self.proj_ess_rej_sub_pop_wrds  = self.exclusive_pop_words_rej(df_text2, f'proj_essay{idx}_cln')

            if len(self.proj_ess_rej_sub_pop_wrds)==0:
                raise Exception('Object not fitted yet!')
            # Step2: Creating feature (Limiting rejected vocabulary to the top 10000 most popular words present in rejected submissions)
            df_text2[f'ess{idx}_rej_scr'] = df_text2.progress_apply(lambda x: self.adv_feat_rej_scr(x, f'proj_essay{idx}_cln', self.proj_ess_rej_sub_pop_wrds.iloc[:10000].index), axis=1)
            print('Generating Rejection Score Done!')

            
            # Dropping unnecessary columns
            df_text2.drop([col, 'proj_essay_pp1', 'proj_essay_pp2', 'project_is_approved'], axis=1, inplace=True)
            
            return df_text2
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)  



    def adv_feat_essay_similarity(self, row):
        
        try: 
            features = [0.0]*10
            SAFE_DIV = 0.001
            
            # essays have stop words removed and stemming is done
            ess1 = row['proj_essay1_cln']
            ess2 = row['proj_essay2_cln']
            
            if len(ess1)==0 or len(ess2)==0:
                return features
            
            
            # Get Unique Words
            e1_words = set(ess1.split())
            e2_words = set(ess2.split())
            
            # Unique Word count
            e1_word_cnt = len(e1_words)
            e2_word_cnt = len(e2_words)
            
            # Unique Common words
            e1_e2_comm_words = len(e1_words.intersection(e2_words))
            
            # Absolute difference b/w word counts in essay1 & essay2
            e1_e2_diff_words = abs(len(ess1.split()) - len(ess2.split()))
            
            # Average length of word counts in essay1 & essay2
            e1_e2_avg_words = (len(ess1.split()) + len(ess2.split()))/2
            
            # Longest substring length
            long_sub = list(distance.lcsubstrings(ess1, ess2))[0]
            e1_e2_long_sub = len(long_sub)
            
            features[0] = e1_e2_comm_words/(min(e1_word_cnt, e2_word_cnt) + SAFE_DIV) # Ratio: common_words_len/min(ess1_len, ess2_len)
            features[1] = e1_e2_comm_words/(max(e1_word_cnt, e2_word_cnt) + SAFE_DIV) # Ratio: common_words_len/max(ess1_len, ess2_len)
            features[2] = e1_e2_diff_words # Absolute difference b/w word counts of ess1 & ess2
            features[3] = e1_e2_avg_words # Average essay length
            features[4] = e1_e2_long_sub/(min(len(ess1), len(ess2)) + SAFE_DIV) # Ratio: longest_substring_len/min(ess1_len, ess2_len)
            features[5] = fuzz.QRatio(ess1, ess2) # fuzz_Qratio
            features[6] = fuzz.WRatio(ess1, ess2) # fuzz_Wratio
            features[7] = fuzz.partial_ratio(ess1, ess2) # fuzz_partial_ratio
            features[8] = fuzz.token_sort_ratio(ess1, ess2) # token_sort_ratio
            features[9] = fuzz.token_set_ratio(ess1, ess2) # token_set_ratio
            
            return features
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)



    def gen_nlp_basic_ess_similar_feats(self, df_ess1:pd.DataFrame, df_ess2:pd.DataFrame, test_mode=0):
        
        '''
        This function extracts the below nlp similarity features from the cleaned textual features: 'essay1' & 'essay2'
        - 'ess_sim_cwc_min'         : Ratio common_words_len/min(ess1_len, ess2_len)
        - 'ess_sim_cwc_max'         : Ratio common_words_len/max(ess1_len, ess2_len)
        - 'ess_sim_dif_word'        : Absolute difference b/w word counts of ess1 & ess2
        - 'ess_sim_avg_word'        : Average essay length
        - 'ess_sim_long_substr_r'   : Ratio longest_substring_len/min(ess1_len, ess2_len)
        
        Below features are using FuzzyWuzzy Library
        - 'ess_sim_fuzz_qr'         : fuzz_Qratio
        - 'ess_sim_fuzz_wr'         : fuzz_Wratio
        - 'ess_sim_fuzz_partial_r'  : fuzz_partial_ratio
        - 'ess_sim_fuzz_tkn_sort_r' : token_sort_ratio
        - 'ess_sim_fuzz_tkn_set_r'  : token_set_ratio

        
        Inputs: 
        - df_essay1 : containing cleaned version of essay1
        - df_essay2 : containing cleaned version of essay2
        
        Helper Functions used:
        - adv_feat_essay_similarity
        
        '''
        
        try: 
            # Merging 2 dataframes
            df_ess = pd.merge(df_ess1, df_ess2, on='id', how='inner')
            
            if test_mode==1:
                df_ess = df_ess.iloc[:1000].copy()
                
            df_text3 = df_ess[['id', 'proj_essay1_cln', 'proj_essay2_cln']].copy()
            
            print('Generating NLP-Basic features: essay similarity')
            # Generating the similarity features b/2 2 essays
            proj_essay_similarity_features = df_text3.progress_apply(self.adv_feat_essay_similarity, axis=1)
            print('Generating Essay Similarity Features Done!')
            
            df_text3['ess_sim_cwc_min'] = list(map(lambda x: x[0], proj_essay_similarity_features))
            df_text3['ess_sim_cwc_max'] = list(map(lambda x: x[1], proj_essay_similarity_features))
            df_text3['ess_sim_dif_word'] = list(map(lambda x: x[2], proj_essay_similarity_features))
            df_text3['ess_sim_avg_word'] = list(map(lambda x: x[3], proj_essay_similarity_features))
            df_text3['ess_sim_long_substr_r'] = list(map(lambda x: x[4], proj_essay_similarity_features))
            df_text3['ess_sim_fuzz_qr'] = list(map(lambda x: x[5], proj_essay_similarity_features))
            df_text3['ess_sim_fuzz_wr'] = list(map(lambda x: x[6], proj_essay_similarity_features))
            df_text3['ess_sim_fuzz_partial_r'] = list(map(lambda x: x[7], proj_essay_similarity_features))
            df_text3['ess_sim_fuzz_tkn_sort_r'] = list(map(lambda x: x[8], proj_essay_similarity_features))
            df_text3['ess_sim_fuzz_tkn_set_r'] = list(map(lambda x: x[9], proj_essay_similarity_features)) 
            
            df_text3.drop(['proj_essay1_cln', 'proj_essay2_cln'], axis=1, inplace=True)
            
            return df_text3  
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            print(custom_exception)



if __name__=='__main__':

    print('Feature Engineering NLP-Basic started: Project Essay!')

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

    # # train_non_text_df = data_clean_obj.clean_nontext_feat(train_df)
    # # test_non_text_df = data_clean_obj.clean_nontext_feat(test_df)
    # # print('Non-Text DF:')
    # # print(train_non_text_df.shape, test_non_text_df.shape)
    # # print(train_non_text_df.columns)
    # # print('-'*print_sep_len)

    # train_text_df = data_clean_obj.clean_text_feat(train_df)
    # test_text_df = data_clean_obj.clean_text_feat(test_df)
    # print('Text DF:')
    # print(train_text_df.shape, test_text_df.shape)
    # print(train_text_df.columns)
    # print('-'*print_sep_len)

    # # train_res_df = data_clean_obj.clean_res(df_inp=train_res_df, fit=1)
    # # test_res_df = data_clean_obj.clean_res(df_inp=test_res_df, fit=0)
    # # print('Resource DF:')
    # # print(train_res_df.shape, test_res_df.shape)
    # # print(train_res_df.columns)
    # # print('-'*print_sep_len)


    # fe_nlp_basic_ess1_obj = FeatureEngineeringNLPBasic2()
    # train_ess1_df = fe_nlp_basic_ess1_obj.gen_nlp_basic_ess_feats(df_inp=train_text_df, col='proj_essay1', fit=1, test_mode=0)
    # test_ess1_df = fe_nlp_basic_ess1_obj.gen_nlp_basic_ess_feats(df_inp=test_text_df, col='proj_essay1', fit=0, test_mode=0)

    # print('-'*print_sep_len)
    # print('NLP-Basic DF: Essay1')
    # print(train_ess1_df.shape)
    # print(train_ess1_df.columns)
    # print(train_ess1_df.isna().sum())
    # print('-'*print_sep_len)
    # print(test_ess1_df.shape)
    # print(test_ess1_df.columns)
    # print(test_ess1_df.isna().sum())


    # fe_nlp_basic_ess2_obj = FeatureEngineeringNLPBasic2()
    # train_ess2_df = fe_nlp_basic_ess2_obj.gen_nlp_basic_ess_feats(df_inp=train_text_df, col='proj_essay2', fit=1, test_mode=0)
    # test_ess2_df = fe_nlp_basic_ess2_obj.gen_nlp_basic_ess_feats(df_inp=test_text_df, col='proj_essay2', fit=0, test_mode=0)

    # print('-'*print_sep_len)
    # print('NLP-Basic DF: Essay2')
    # print(train_ess2_df.shape)
    # print(train_ess2_df.columns)
    # print(train_ess2_df.isna().sum())
    # print('-'*print_sep_len)
    # print(test_ess2_df.shape)
    # print(test_ess2_df.columns)
    # print(test_ess2_df.isna().sum())


    # fe_nlp_basic_ess_sim_obj = FeatureEngineeringNLPBasic2()
    # train_ess_sim_df = fe_nlp_basic_ess_sim_obj.gen_nlp_basic_ess_similar_feats(df_ess1=train_ess1_df, df_ess2=train_ess2_df, test_mode=0)
    # test_ess_sim_df = fe_nlp_basic_ess_sim_obj.gen_nlp_basic_ess_similar_feats(df_ess1=test_ess1_df, df_ess2=test_ess2_df, test_mode=0)

    # print('-'*print_sep_len)
    # print('NLP-Basic DF: Essay2')
    # print(train_ess_sim_df.shape)
    # print(train_ess_sim_df.columns)
    # print(train_ess_sim_df.isna().sum())
    # print('-'*print_sep_len)
    # print(test_ess_sim_df.shape)
    # print(test_ess_sim_df.columns)
    # print(test_ess_sim_df.isna().sum())

    


    
