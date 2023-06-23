#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pandas import read_sql
import subprocess
import sys
import pg8000 as dbapi
import datetime as dt
from datetime import date, datetime, timedelta
import psycopg2
from secure_ai_sandbox_python_lib.session import Session
import boto3
import os
import io
from io import StringIO
import math
import numpy as np
import multiprocessing.pool
from time import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from multiprocessing import Pool


def reader(filename):
    """Read in categorical feature raw data, skip erroneous rows"""
    return pd.read_csv(os.path.join(LOCAL_DOWNLOAD_PATH,filename),delimiter=",", engine='python',quotechar='"', error_bad_lines=False)

def encode_cat(raw, filepath1, filepath2):
    """Encode categorical features with risk table, risk table is produced with MOIndex_risktable.py"""
    """raw: input file with raw features
        filepath1: file path for categoricalVariableValsToRepresent.json
        filepath2: file path for categoricalVariableValsCountAndFraudRate.json"""
    f = open(filepath1)

    cat = json.load(f)
    cat = [k for k in cat]

    f = open(filepath2)
    data = json.load(f)

    for i in data:
        print(i)
        current = pd.DataFrame(data[i]).T.reset_index()[['index','r']]
        current = current.rename(columns={'index':i})
        unseen = current.loc[current[i]=='unseen']['r'].unique()[0]
        tmp = raw.set_index(i).join(current.set_index(i),on=i,how='left').reset_index()
        tmp.loc[tmp.r.isnull(),'r'] = unseen
        tmp[i] = tmp['r']
        tmp = tmp.drop(['r'],axis=1)
        processed = tmp
    return processed

def run_sql(query, con = conn):
    """Py passthrough to run sql query against redshift table"""
    """query: sql query to run
        con: connection variables"""
    df = read_sql(query, con= con)
    df.columns = [col for col in df.columns.values]
    return df

def load_in_feat_file_names(awsAccessKeyId,awsSecretKey,bucket,matching_string):
    """Read in file names from S3 based on file name pattern matching"""
    """awsAccessKeyId: aws Access Key Id
        awsSecretKey: aws Secret Key
        bucket: bucket name
        matching_string: file name pattern to match"""
    s3_client = boto3.client("s3", aws_access_key_id = awsAccessKeyId, aws_secret_access_key = awsSecretKey)
    s3_resource = boto3.resource('s3',aws_access_key_id = awsAccessKeyId, aws_secret_access_key = awsSecretKey)

    session = boto3.Session(
         aws_access_key_id=awsAccessKeyId,
         aws_secret_access_key=awsSecretKey)

    s3 = session.resource('s3')
    s3_bucket = s3.Bucket(bucket)

    file_names = []

    for my_bucket_object in s3_bucket.objects.all():
        if my_bucket_object.key.startswith(matching_string+'{}.csv/'.format(date)):
            file_names.append(my_bucket_object.key)

    print('file #:',len(file_names))
    return file_names

def read_in_local_files(matching_string):
    """Read designated local files into a dataframe based on file name pattern matching"""
    """matching_string: file name pattern to match"""
    feats = pd.DataFrame()
    for f in os.listdir(LOCAL_DOWNLOAD_PATH):
        if f.startswith(matching_string+date):
            current = pd.read_csv(LOCAL_DOWNLOAD_PATH+f)
            feats = pd.concat([feats,current],axis=0,ignore_index=True)
    feats.columns = [i.lower() for i in feats.columns.tolist() ]
    return feats

def add_key(df):
    """Convert date from string to datetime, add key = sessionid + date"""
    """df: input dataframe"""
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df['date'] = df['date'].apply(lambda x: x[:-9] if type(x) == str else x)
    df['key'] = df['sessionid'] + '-' + df['date']
    df = df.loc[df.date == date]
    return df

def get_f_n_r(data, cat, data_dir=LOCAL_DOWNLOAD_PATH, cate_subdir = 'cat_vars', fraud_col = 'tag', alpha = alpha ):
    """
    for each categorical variable var, collect
    val (including empty value '')
    counts
    good_counts(baed on fraud col = 0)
    bad_counts(based on fraud col = 1)
    frac (in all available values)
    cumsum frac(cumulative sum fraction)
    fraud_rate (based on fraud col 0 and 1 ONLY)
    sort by counts
    write to file cat_vars
    unseen val -> alpha prior
    """
    t0 = time()
    for var in cat:
        print("Processing variable: " + var)
        if fraud_col not in data.columns:
            raise Exception('fraud_col', fraud_col,  'does NOT exist!!!')
        if var not in data.columns:
            raise Exception('var', var,  'does NOT exist!!!')
        na_key = '' #define empty value to be saved in risk table map
        data[var].fillna(na_key, inplace=True)
        df = data[var].value_counts(ascending=False).to_frame().reset_index()
        df.columns = ['value', 'counts']
        good_counts =  data[(~pd.isna(data[fraud_col]) & (data[fraud_col] == 0))][var].value_counts().to_frame().reset_index()
        good_counts.columns = ['value', 'good_counts']
        bad_counts = data[(~pd.isna(data[fraud_col]) & (data[fraud_col] == 1))][var].value_counts().to_frame().reset_index()
        bad_counts.columns = ['value', 'bad_counts']
        df = df.merge(good_counts, how='left', on='value', copy=False)
        df = df.merge(bad_counts, how='left', on='value', copy=False)
        df.fillna(0, inplace=True)

        total_counts = df['counts'].sum()
        df['frac'] = df['counts'] / total_counts
        df['cumsum_frac'] = df['frac'].cumsum()
    # for values without any confirmed is_fraud value 0, 1, fill the fraud rate as 0
        df['fraud_rate'] = (df['bad_counts'] / (df['good_counts'] + df['bad_counts'])).fillna(0)
        output_dir = os.path.join(data_dir, cate_subdir)
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        except Exception as e:
            print('some issue creating the folder', e)

    #add unseen
        df = df.append({'value':'unseen','counts':0,'good_counts':0,'bad_counts':0,'frac':0,'cumsum_frac':1,'fraud_rate':alpha},ignore_index=True)
        df.to_csv(os.path.join(output_dir, var+ '.csv'), index =False)
        del good_counts
        del bad_counts
        del df
        print(var, 'saved in', output_dir)
        tt = time() - t0
        print("Completed in {} seconds ".format(round(tt,3)) + " for variable " + var)

def read_categ_csv(var:str, cate_subdir, na_value, LOCAL_DOWNLOAD_PATH):
    """read existing value metric, value as str, and replace emtpy value with preset na value"""
    """
            val_metric columns arranged as follows:
            0: value; 1: counts; 2: good_counts; 3: bad_counts; 4: frac; 5: cumsum_frac; 6: fraud_rate
    """

    val_metric = pd.read_csv(os.path.join(LOCAL_DOWNLOAD_PATH, cate_subdir, var+'.csv'), dtype = {'value':object})
        # fill null value of 'val' as empty string ''
    val_metric['value'] = val_metric['value'].fillna(na_value)
    return val_metric

def categorical_helper(var:str, high_card_freq_pct: float=2.0, minimum_freq: int = 5, top_n_freq: int = 20,
                      top_n_fraudrate: int=20, top_cumsum: float=0.8, max_n_distinct_val_cumsum: int=50,
                      cate_lst = cat):
    """Apply a number of filters to values for each cat feat"""
    getVal = itemgetter(0)
    t0 = time()
    try:
        val_metric = read_categ_csv(var=var, cate_subdir='cat_vars', na_value='')
        print('finished reading', var)
        val_metric = [tuple(x) for x in val_metric.values]
        max_freq_frac=max([i[4] for i in val_metric]) #i[4] is frac
        if max_freq_frac*100 <high_card_freq_pct: #if for a cat feat, the highest freq value's frac is < 0.05, remove the feat
            print('### Please remove high card and even-freq variable : ', var)
            cate_lst.remove(var)
            print('!!!removed {} from cate_lst'.format(var))

        # top values sort by counts if minimum frequency satisfies filter, else retain original sort
        freq_lst = list(map(getVal, sorted(val_metric, key = lambda x: x[1] if x[1] > minimum_freq else 0, reverse=True)[: top_n_freq])) #x[1] is count, get top n frequency feat values, they should all be greater than min freq threashold
        print('freq_lst:',len(freq_lst))
        # top values sort by fraud rate if minimum frequency satisfies filter, else retain original sort
        rate_lst = list(map(getVal, sorted(val_metric, key = lambda x: x[6] if x[1] > minimum_freq else 0, reverse=True)[: top_n_fraudrate])) #x[6] is mo_rate, get top n mo_rate feat values, they should all be greater than min freq threashold
        print('rate_lst:',len(rate_lst))
        cumsum_lst = []
        if top_cumsum > 0 and top_cumsum < 1.0:
            cumsum_sort = sorted(val_metric, key =lambda x: x[5]) #x[5] is cumsum frac
            index = 0
            while cumsum_sort[index][5] < top_cumsum and index < len(cumsum_sort):
                cumsum_lst.append(cumsum_sort[index][0])
                index += 1
            if index < len(cumsum_sort):
                cumsum_lst.append(cumsum_sort[index][0])
        cumsum_lst = cumsum_lst[:max_n_distinct_val_cumsum] #get top n feat vals that dominant the top 60% cumsum frac
        print('cumsum_lst:',len(cumsum_lst))
        # combined as a set, sort for reproducibility
        unique_vals = sorted(list(set(freq_lst + rate_lst + cumsum_lst + ['unseen'])))
        print(f'num of top values to be saved for {var} : {len(unique_vals)} ')
        return unique_vals
    except:
        print('failed to find representative value for variable ', var)


def categorical_combine(outputValsJSON: str, multithread = 3,
                            minimum_freq=5, high_card_freq_pct= 2.0,
                            top_n_freq=20, top_n_fraudrate=20,
                            top_cumsum= 0.8, max_n_distinct_val_cumsum= 50, cate_lst = cat):
    """Select qualified feat values to save into risk table"""
    def _func(var):
        return categorical_helper(var = var, high_card_freq_pct=high_card_freq_pct, minimum_freq=minimum_freq,
                                           top_n_freq=top_n_freq, top_n_fraudrate=top_n_fraudrate,top_cumsum = top_cumsum,
                                           max_n_distinct_val_cumsum = max_n_distinct_val_cumsum)
    with multiprocessing.pool.ThreadPool(multithread) as pool:
        start=time()
        top_value_lsts = list(pool.map(_func, cate_lst))
        end=time()
    top_value_lsts = [vals for vals in top_value_lsts if vals is not None]
    var_topval_dict = {x: y for x, y in zip(cate_lst, top_value_lsts) if y is not None}
    print(var_topval_dict)
    try:
        with open(os.path.join(LOCAL_DOWNLOAD_PATH, outputValsJSON), 'w') as fp:
            json.dump(var_topval_dict, fp)
            print('Saved top values in ', outputValsJSON)
        print('finished in ', end -start, 's')
    except:
        print('Problem saving selectedVar json')


def risk_table(outputRiskTableJSON: str, multithread: int,
                    cate_lst = cat,
              LOCAL_DOWNLOAD_PATH):
    """
        save out dict that first keyed by categorical variable
        secondary keyed by possible values
        third level key 'r'and 'f':
            r(fraud rate, aka [6] in csv ) f(frequency, aka 'counts', [1] in the csv)
        {cate_var1: {val1: {"r": float, "f": int}, val2: {}...}, cate_var2: {} ...}
    """
    def risk_table_helper(var: str):
        """
            for each var, convert interested field into dictionary
        """
        val_metric = read_categ_csv(var=var,cate_subdir='cat_vars', na_value='')
        f = open(LOCAL_DOWNLOAD_PATH+'categoricalVariableValsToRepresent.json')
        data = json.load(f)
        val_lst = data[var]
        val_metric = [tuple(x) for x in val_metric.values if x[0] in val_lst]
        val_dict = {x[0]: {'f': x[1], 'r': x[6]} for x in val_metric}
        print(val_dict)
        return val_dict

    with multiprocessing.pool.ThreadPool(multithread) as pool:
        start=time()
        var_risktable = list(pool.map(risk_table_helper, cate_lst))
    all_risktable = {x: y for x, y in zip(cate_lst, var_risktable)}
    try:
        with open(os.path.join(LOCAL_DOWNLOAD_PATH, outputRiskTableJSON), 'w') as fp:
            json.dump(all_risktable, fp)
            print('Saved risk table in ', outputRiskTableJSON)
            end=time()
        print('finished in ', end -start, 's')
    except:
        print('Problem saving risk table')
