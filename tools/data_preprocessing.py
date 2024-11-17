'''
Read raw PhysioNet 2012 data, preprocess it and save it in json format.
'''

import os
import json
import argparse
import numpy as np
import pandas as pd


static_vars = ['RecordID', 'Age', 'Gender', 'Height', 'Weight', 'ICUType']
temporal_vars = ['Temp', 'pH', 'FiO2', 'TroponinT', 'Creatinine', 'PaCO2', 'HCT', 'TroponinI',\
    'AST', 'Mg', 'SysABP', 'RespRate', 'NIDiasABP', 'Platelets', 'Cholesterol', 'Albumin', 'MechVent',\
    'NISysABP', 'Glucose', 'MAP', 'ALT', 'Lactate', 'Na', 'K', 'WBC', 'SaO2', 'HCO3', 'Bilirubin',\
    'BUN', 'ALP', 'Weight', 'DiasABP', 'PaO2', 'Urine', 'HR', 'GCS', 'NIMAP']


def split_df(df):
    '''
    Split the dataframe into static and temporal data.
    Args:
        df (pd.DataFrame): dataframe to split
        
    Returns:
        df_static (dict): static data
        df_temporal (dict): temporal data
    '''

    # extract static variables into a separate dataframe
    df_static = df.loc[df['Time'] == '00:00', :].copy()

    # retain only one of the 6 static vars:
    df_static = df_static.loc[df['Parameter'].isin(static_vars)]
    
    # extract temporal variables into a separate dataframe
    idx_temporal = df_static.index
    df_temporal = df.loc[~df.index.isin(idx_temporal), :]
    
    return df_static, df_temporal
    
    
def preprocess_static(dict_static):
    '''
    Preprocess the static data.
    Args:
        dict_static (dict): static data
    
    Returns:
        dict_static (dict): preprocessed static data
    '''
    
    # delete RecordID
    del dict_static['RecordID']
    
    for key, value in dict_static.items():
        # Age
        # Gender
        # Height    
        if key == 'Height':
            if value < 0:
                dict_static[key] = -1
                            
            elif value < 10: # 1.8 -> 180
                dict_static[key] = value * 100
                
            elif value < 25: # 18 -> 180
                dict_static[key] = value * 10
                
            elif value < 100: # 81.8 -> 180 (inch -> cm)
                dict_static[key] = value * 2.54
                
            elif value > 1000: # 1800 -> 180
                dict_static[key] = value * 0.1

            elif value > 250: # 400 -> 157
                dict_static[key] = value * 0.3937
        
        # ICUType
        # Weight 
        elif key == 'Weight':
            if value < 35:
                dict_static[key] = -1
            elif value > 300:
                dict_static[key] = -1
                    
    return dict_static


def preprocess_temporal(df_temporal):
    '''
    Preprocess the temporal data.
    Args:
        df_temporal (pd.DataFrame): temporal data
        
    Returns:
        dict_temporal (dict): preprocessed temporal data
    '''
    
    # Convert time to minutes
    df_temporal.loc[:, 'Time'] = df_temporal['Time'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))

    def delete_value(df, c, value=0):
        idx = df['Parameter'] == c
        idx = idx & (df['Value'] == value)
        
        df.loc[idx, 'Value'] = np.nan
        return df

    def replace_value(df, c, value=np.nan, below=None, above=None):
        idx = df['Parameter'] == c
        
        if below is not None:
            idx = idx & (df['Value'] < below)
            
        if above is not None:
            idx = idx & (df['Value'] > above)
        
        
        if 'function' in str(type(value)):
            # value replacement is a function of the input
            df.loc[idx, 'Value'] = df.loc[idx, 'Value'].apply(value)
        else:
            df.loc[idx, 'Value'] = value
            
        return df
    
    df_temporal = delete_value(df_temporal, 'DiasABP', -1)
    df_temporal = replace_value(df_temporal, 'DiasABP', value=np.nan, below=1)
    df_temporal = replace_value(df_temporal, 'DiasABP', value=np.nan, above=200)
    df_temporal = replace_value(df_temporal, 'SysABP', value=np.nan, below=1)
    df_temporal = replace_value(df_temporal, 'MAP', value=np.nan, below=1)

    df_temporal = replace_value(df_temporal, 'NIDiasABP', value=np.nan, below=1)
    df_temporal = replace_value(df_temporal, 'NISysABP', value=np.nan, below=1)
    df_temporal = replace_value(df_temporal, 'NIMAP', value=np.nan, below=1)

    df_temporal = replace_value(df_temporal, 'HR', value=np.nan, below=1)
    df_temporal = replace_value(df_temporal, 'HR', value=np.nan, above=299)

    df_temporal = replace_value(df_temporal, 'PaCO2', value=np.nan, below=1)
    df_temporal = replace_value(df_temporal, 'PaCO2', value=lambda x: x*10, below=10)

    df_temporal = replace_value(df_temporal, 'PaO2', value=np.nan, below=1)
    df_temporal = replace_value(df_temporal, 'PaO2', value=lambda x: x*10, below=20)

    # the order of these steps matters
    df_temporal = replace_value(df_temporal, 'pH', value=lambda x: x*10, below=0.8, above=0.65)
    df_temporal = replace_value(df_temporal, 'pH', value=lambda x: x*0.1, below=80, above=65)
    df_temporal = replace_value(df_temporal, 'pH', value=lambda x: x*0.01, below=800, above=650)
    df_temporal = replace_value(df_temporal, 'pH', value=np.nan, below=6.5)
    df_temporal = replace_value(df_temporal, 'pH', value=np.nan, above=8.0)

    # convert to Fahrenheit
    df_temporal = replace_value(df_temporal, 'Temp', value=lambda x: x*9/5+32, below=10, above=1)
    df_temporal = replace_value(df_temporal, 'Temp', value=lambda x: (x-32)*5/9, below=113, above=95)

    df_temporal = replace_value(df_temporal, 'Temp', value=np.nan, below=25)
    df_temporal = replace_value(df_temporal, 'Temp', value=np.nan, above=45)

    df_temporal = replace_value(df_temporal, 'RespRate', value=np.nan, below=1)
    df_temporal = replace_value(df_temporal, 'WBC', value=np.nan, below=1)

    df_temporal = replace_value(df_temporal, 'Weight', value=np.nan, below=35)
    df_temporal = replace_value(df_temporal, 'Weight', value=np.nan, above=299)
    
    df_temporal = df_temporal.dropna(subset=['Value'])
    
    dict_temporal = {}
    
    for var in temporal_vars:
        df_var = df_temporal[df_temporal['Parameter'] == var]
        dict_temporal[var] = {
            'Time': df_var['Time'].values.tolist(),
            'Value': df_var['Value'].values.tolist(),
        }
        len_ = len(df_var)
        if len_ > max_lens[var]:
            max_lens[var] = len_
        
    return dict_temporal


def save_json(records, dest_path):
    '''
    Save the records in json format.
    Args:
        records (dict): records to save
        dest_path (str): path to save the records
        
    Returns:
        None
    '''
    
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    with open(os.path.join(dest_path, f'{records["RecordID"]}.json'), 'w') as fp:
        json.dump(records, fp, indent=4)
        fp.close()

    
def preprocess_and_save(df_static, df_temporal, df_target, dest_path):
    '''
    Preprocess the static and temporal data and save it in json format.

    Args:
        df_static (pd.DataFrame): static data
        df_temporal (pd.DataFrame): temporal data
        
    Returns:
        None
    '''
    
    dict_static = df_static.set_index('Parameter')['Value'].to_dict()
    
    ID = int(dict_static['RecordID'])
    
    records = {
        'RecordID': ID,
        'Static': preprocess_static(dict_static),
        'Temporal': preprocess_temporal(df_temporal),
        'Outcome': int(df_target[df_target['RecordID'] == ID]['In-hospital_death'].values[0]),
    }
    
    save_json(records, dest_path)


def run(src_path, dest_path, outcome_path):
    '''
    Read data in the given path, preprocess it and save it in json format.

    Args:
        src_path (str): path to the raw dataset
        dest_path (str): path to save the preprocessed data
        outcome_path (str): path to the outcome file
        
    Returns:
        None 
    '''
    
    # Create destination directory if it does not exist
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    # Traverse through all records
    dirs = os.listdir(src_path)
    assert len(dirs) == 4000, 'Number of records is 4000'
    
    df_target = pd.read_csv(outcome_path, sep=',')
    
    global max_lens
    max_lens = {var: 0 for var in temporal_vars}

    for d in dirs:
        # read the record
        df = pd.read_csv(os.path.join(src_path, d))
        df_static, df_temporal = split_df(df)
        preprocess_and_save(df_static, df_temporal, df_target, dest_path)
    
    # save the max lengths to csv file
    with open(os.path.join(dest_path, 'max_lens.csv'), 'w') as fp:
        for key, value in max_lens.items():
            fp.write(f'{key},{value}\n')
        fp.close()
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./datasets/PysioNet2012/', help='path to the dataset directory')
    parser.add_argument('--set', type=str, default='set-a', help='subset to preprocess')
    parser.add_argument('--output', type=str, default='./datasets/PysioNet2012_Preprocessed/', help='path to save the preprocessed data')
    
    args = parser.parse_args()
    
    # create destination directory if it does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    # outcome path
    set_name = args.set.split('-')[-1]
    outcome_path = os.path.join(args.path, f'Outcomes-{set_name}.txt')
    
    run(
        src_path=os.path.join(args.path, args.set),
        dest_path=os.path.join(args.output, args.set),
        outcome_path=outcome_path,
    )
    