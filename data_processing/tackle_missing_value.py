import os
import pandas as pd



def fill_not_applicable(df: pd.DataFrame):
    ''' 
    Certain variables are inherently associated with high missing rates,
    due to their definition and scope.
    For these variables, missing values correspond to cases where the variable is 

    “Not applicable” 

    to a given patient, rather than cases of unobserved or corrupted data.
    '''
    df.loc[(df['DETCRIM'] == -9) & (df['PSOURCE'] != 7), 'DETCRIM'] = 0
    df.loc[(df['DETNLF'] == -9) & (df['EMPLOY'] != 4), 'DETNLF'] = 0
    df.loc[(df['DETNLF_D'] == -9) & (df['EMPLOY_D'] != 4), 'DETNLF_D'] = 0
    df.loc[(df['PREG'] == -9) & (df['GENDER'] != 2), 'PREG'] = 0
    return df

def _fill_help(df:pd.DataFrame, sub, target_var):
    df.loc[(df[sub] == 1) & (df[target_var] == -9), target_var] = 0
    return df

def fill_not_available(df:pd.DataFrame):
    '''
    Another major source of missing values arises from inter-variable dependencies 
    in clinical documentation. In many cases, the recording of a downstream variable 
    depends on the presence or value of an upstream variable. 
    
    If the upstream condition is not met, the downstream variable is recorded as unavailable.
    Such missingness does not indicate the absence of information but rather reflects 
    a procedural decision in the data collection process. 
    
    In the dataset, these cases are typically labeled as 
    
    “Unavailable” 
    
    and are encoded as missing values in the raw format.
    '''
    variables = ('SUB', 'FREQ', 'ROUTE', 'FRSTUSE')
    for i in ('1', '2', '3'):
        cur_var = [name + i for name in variables]

        # admission variable
        df = _fill_help(df, cur_var[0], cur_var[1])
        df = _fill_help(df, cur_var[0], cur_var[2])
        df = _fill_help(df, cur_var[0], cur_var[3])

        # discharge variable '_D'
        df = _fill_help(df, cur_var[0] + '_D', cur_var[1] + '_D')
    return df

def tackle_missing_value(raw_df_path: str):
    raw_df = pd.read_csv(raw_df_path)
    missing_corrected = fill_not_applicable(raw_df)
    missing_corrected = fill_not_available(missing_corrected)
    return missing_corrected

def tackle_missing_value_main(raw_df_path: str, missing_corrected_path: str):
    if os.path.exists(missing_corrected_path):
        missing_corrected = pd.read_csv(missing_corrected_path)
        return missing_corrected
    return tackle_missing_value(raw_df_path)