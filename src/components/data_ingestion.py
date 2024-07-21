import sys
import os
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split



class DataIngest:

    def __init__(self):
        self.train_data_path = os.path.join('artifacts', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'test.csv')
        self.train_data_res_path = os.path.join('artifacts', 'train_res.csv')
        self.test_data_res_path = os.path.join('artifacts', 'test_res.csv')
        

    def start_data_ingestion_from_csv(self, sample_size=0.15, test_size=0.25, random_state=42):

        try:
            # Reading the complete csv files
            data = pd.read_csv(os.path.join('notebooks/data/model_building/data.csv'))
            data_res = pd.read_csv(os.path.join('notebooks/data/model_building/resources.csv'))
            
            # Sampling data
            sample_row_count = int(sample_size*data.shape[0])
            data_sample = data.sample(n=sample_row_count, random_state=random_state)


            # Creating train-test split for data
            train_data, test_data = train_test_split(data_sample, test_size=test_size, random_state=random_state)

            # Creating train-test split for resources
            train_proj_ids = train_data['id'].values
            train_data_res = data_res.loc[data_res['id'].isin(train_proj_ids)].copy()
            test_data_res = data_res.loc[~data_res['id'].isin(train_proj_ids)].copy()


            # Creating a directory
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)

            # Saving the dfs:
            train_data.to_csv(self.train_data_path, sep=',', index=False)
            test_data.to_csv(self.test_data_path, sep=',', index=False)
            train_data_res.to_csv(self.train_data_res_path, sep=',', index=False)
            test_data_res.to_csv(self.test_data_res_path, sep=',', index=False)

            return self.train_data_path, self.test_data_path, self.train_data_res_path, self.test_data_res_path,

        except Exception as e: 
            custom_exception = CustomException(e, sys)
            print(custom_exception)


if __name__=='__main__':

    print('Ingestion Testing started!')
    ingest_obj = DataIngest()
    train_pt, test_pt, train_res_pt, test_res_pt = ingest_obj.start_data_ingestion_from_csv(sample_size=0.15, test_size=0.25, random_state=42)

    print(train_pt)
    print(test_pt)
    print(train_res_pt)
    print(test_res_pt)




