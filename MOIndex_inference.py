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
import pickle as pkl
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import tensorflow as tf
from utils.py import reader, encode_cat, run_sql, load_in_feat_file_names, read_in_local_files, add_key


###################Define variables####################################

print('starting job ',date,' at ',date, datetime.now().strftime("%H:%M:%S"))
date = date.today() - timedelta(days=1)
date = date.strftime("%Y-%m-%d")
print('date:',date)
print('starting ',date,' job at',date,datetime.now().strftime("%H:%M:%S"))

LOCAL_DOWNLOAD_PATH = "home/guanxian/"
awsAccessKeyId = 'xyz'
awsSecretKey = '123'
bucket = 'moindex'


####################Load in feature names####################################
# load in final list of features
all_feat = pd.read_csv('/home/guanxian/final_feat.csv',sep=',')
feat = all_feat.columns.tolist()

cat_feat = feat[:27].tolist()
num_feat = [i for i in feat if i not in cat_feat]

all_feat = cat_feat + num_feat


###################Load in numerical features raw data from S3####################

file_names = load_in_feat_file_names(awsAccessKeyId,awsSecretKey,bucket,'num_feat_daily_')

#download in local dir
KEYS_TO_DOWNLOAD = file_names

def download_object(file_name):
    """Downloads an object from S3 to local."""

    download_path = LOCAL_DOWNLOAD_PATH + file_name[25:53]+'_'+file_name[63:68]+'.csv'

    print(f"Downloading {file_name} to {download_path}")
    s3_client.download_file(
        bucket,
        file_name,
        str(download_path)
    )
    return "Success"

def download_parallel_multiprocessing():
    """Download objects in parallel"""
    with ProcessPoolExecutor() as executor:
        future_to_key = {executor.submit(download_object, key): key for key in KEYS_TO_DOWNLOAD}

        for future in futures.as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()

            if not exception:
                yield key, future.result()
            else:
                yield key, exception

if __name__ == "__main__":
    for key, result in download_parallel_multiprocessing():
        print(f"{key}: {result}")

num_feats = read_in_local_files('num_feat_daily_')
print('read in numerical feat, preprocess')


################Process numerical features######################################

#convert date from string to datetime, add key
num_feats = add_key(num_feats)

#_200_vec is a 128 dimension vector feature saved as string, this step is to convert string to 128 numerical features
num_feats._200_vec = num_feats._200_vec.apply(lambda x: x.split(' ') if type(x)!=float else [])
columns = []
for i in range(128):
    columns.append('_200_vec_{}'.format(i))

num_feats[columns] = pd.DataFrame(num_feats._200_vec.tolist(), index= num_feats.index)
num_feats = num_feats.drop(['_200_vec'],axis=1)

#because 1 session may span multiple timestamps, resulting in multiple rows per key, this step is to consolidate
#records to produce 1 record per key (session+date) with mean values of each numerical feature
num_feats[num_feat] = num_feats[num_feat].astype(float)
num_feats = num_feats.groupby(['key','sessionid','date']).mean().reset_index()


################## Load in categorical features raw data from S3####################

file_names = load_in_feat_file_names(awsAccessKeyId,awsSecretKey,bucket,'cat_feat_daily_')

#download in local dir
KEYS_TO_DOWNLOAD = file_names

def download_object(file_name):
    """Downloads an object from S3 to local."""

    download_path = LOCAL_DOWNLOAD_PATH + file_name[25:53]+'_'+file_name[63:68]+'.csv'

    print(f"Downloading {file_name} to {download_path}")
    s3_client.download_file(
        bucket,
        file_name,
        str(download_path)
    )
    return "Success"

def download_parallel_multiprocessing():
    """Download objects in parallel"""
    with ProcessPoolExecutor() as executor:
        future_to_key = {executor.submit(download_object, key): key for key in KEYS_TO_DOWNLOAD}

        for future in futures.as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()

            if not exception:
                yield key, future.result()
            else:
                yield key, exception

if __name__ == "__main__":
    for key, result in download_parallel_multiprocessing():
        print(f"{key}: {result}")

#load in cat feats
cat_feats = read_in_local_files('cat_feat_daily_')
print('read in categorical feat, preprocess')


######################Process categorical features#################################

#convert date, add key
cat_feats = add_key(cat_feats)

#only keeping features that are in the final feature list
cat_feats = cat_feats[cat_feat]

#because 1 session may span multiple timestamps, resulting in multiple rows per key, this step is to consolidate #records to produce 1 record per key (session+date) with mode values of each categorical feature
tmp = cat_feats[['key']].drop_duplicates()
for i in cat_feat:
    print(i)
    cat_feats[i] = cat_feats[i].fillna('')
    bc_bes_cat_feat_tmp = cat_feats.groupby(['key'])[i].apply(lambda x: x.mode().iloc[0]).reset_index() #find mode of col by key, in case of multiple
    bc_bes_cat_feat_tmp.columns = ['key',i]
    tmp = tmp.set_index(['key']).join(bc_bes_cat_feat_tmp.set_index(['key']),on=['key'],how='left').reset_index()
    print(tmp.head(4))
    cat_feats = tmp
    cat_feats.columns = [i+'_raw' if i != 'key' else i for i in cat_feats.columns]cat_feats.columns = [i[:-4] if i[-4:] == '_raw' else i for i in cat_feats.columns]


#save processed categorical features in case of deep dive
cat_raw = cat_feats
cat_raw.to_csv(LOCAL_DOWNLOAD_PATH+'cat_raw_{}.csv'.format(date))
cat_raw.shape


#encode cat feat with risk table
cat_feats[cat_feat] = cat_feats[cat_feat].astype(str)
cat_feats[cat_feat] = encode_cat(cat_feats[cat_feat], '/home/guanxian/categoricalVariableValsToRepresent.json','home/guanxian/categoricalVariableValsCountAndFraudRate.json')

##join num and cat feats
print('join num and cat feats')
all_feats = num_feats.set_index('key').join(cat_feats.set_index('key'),how='left',on='key',lsuffix='_bes',rsuffix='_cat').reset_index()
print('all_feats shape:',all_feats.shape)
print('all_feats keys:',all_feats.key.nunique())
print('all_feats sessions',all_feats.sessionid.nunique())


#only keep necessary cols
all_feats_ = all_feats[all_feat]


##################### Load in model artifacts to run inference####################################

scaler = pkl.load(open(LOCAL_DOWNLOAD_PATH+'minmax_scaler', 'rb'))
model_ = pkl.load(open(LOCAL_DOWNLOAD_PATH+'ae_model', 'rb'))
with open(LOCAL_DOWNLOAD_PATH+"ae_kmeans.pkl", "rb") as f:
    kmeans_model = pickle.load(f)
bad_cluster_list = pd.read_csv(LOCAL_DOWNLOAD_PATH+'bad_clusters_.csv')
bad_cluster_list = bad_cluster_list.columns.tolist()
bad_cluster_list = [int(x) for x in bad_cluster_list]
distance_thr = pd.read_csv(LOCAL_DOWNLOAD_PATH+'distance_thr_.csv')
distance_thr = distance_thr.dist_thr.iloc[0]

#impute NA, apply scaler and mlp model to get embedding layer
all_feats_ = all_feats_.fillna(-1)
all_feats_ = scaler.transform(all_feats_)
all_feats2 = model_.predict(all_feats_)

#apply kmeans model, get cluster membership and distance to cluster centers
all_feats_embed = pd.DataFrame(all_feats2)
all_feats_embed['cluster'] = kmeans_model.predict(all_feats2)
all_feats_embed['distance'] = np.min(kmeans_model.transform(all_feats2),axis=1)

#select sessions that are in the 'high purity bad clusters' and whose distance to cluster centers are less than the guardrail distance value
all_feats_embed_selected = all_feats_embed.loc[(all_feats_embed.cluster.isin(bad_cluster_list)) & (all_feats_embed.distance <= distance_thr)]

#derive high risk sessions' key, sessionid and date, construct final output dataframe with key, sessionid, date, cluster membership and distance to centroids
t1 = all_feats.loc[all_feats_.index.isin(all_feats_embed_selected.index)][['key','sessionid','date']]
all_feats = pd.concat([t1,all_feats_embed_selected[['distance']]],axis=1)


#############################Derive associating customerids, remove those that are already enforced, output final list of customerids, upload to EDX for auto enforcement

#derive customerids corresponding to high risk sessions from redshift
start_date = date
end_date = date
session = Session('.')
sts = boto3.client('sts')
creds = sts.assume_role(RoleArn="myarn",
                        # The role used to get Redshift credentials, this is unique to the account the cluster is in
                         RoleSessionName="my_session",
                         Tags=[
                            {
                                'Key': 'currentOwnerAlias',
                                'Value': session.owner_alias()
                            },
                        ],
                        TransitiveTagKeys=["currentOwnerAlias"])["Credentials"]
# Connect
redshift_boto3 = boto3.client(
    'redshift',
    aws_access_key_id=creds['AccessKeyId'],
    aws_secret_access_key=creds['SecretAccessKey'],
    aws_session_token=creds['SessionToken'],
)
dbname= 'db'
redshift_login = redshift_boto3.get_cluster_credentials( DbUser= session.owner_alias() + '_sais',
    DbName=dbname,
    ClusterIdentifier='dbid',
    AutoCreate=True,
    DbGroups=['guanxian'],
    DurationSeconds=3600)
conn = dbapi.connect(database=dbname,
                          host='host',
                          port=1234,
                          user=redshift_login['DbUser'],
                          password=redshift_login['DbPassword'],
                          ssl_context=True)

CIDs = run_sql("""select
distinct sessionid, customerid, date
from db.my_table where date_trunc('day',date) >= '{}' and date_trunc('day',date) <= '{}'
and customerid is not null and customerid <> '0'""".format(start_date, end_date))

CIDs.date = CIDs.date.dt.strftime('%Y-%m-%d')
CIDs['key'] = CIDs['sessionid']  +'-'+ CIDs['date']
CIDs.head(4)

#join CIDs to model output on sessionid
selected_cids = all_feats.merge(CIDs[['key','customerid']], on='key', how='inner').sort_values(['customerid'],ascending=False)

#filter out customerids that have already been enforced
cids = selected_cids.customerid.unique()
check_duplicates = run_sql("""create temp table base as
select distinct customerid, detection_date
from db.my_table1
where customerid in ("""+','.join(str(x) for x in cids)+""") and detection_date >= '"""+date+"""'
union all
(select distinct customer_id as customerid, date as detection_date
            from db.my_table2
            where risk_level = 'CRITICAL'
            and date >= '"""+date+"""'
            and customerid in ("""+','.join(str(x) for x in cids)+"""))
union all
(select distinct customerid, date as detection_date
            from db.my_table3
            where date >= '"""+date+"""'
            and customerid in ("""+','.join(str(x) for x in cids)+"""))
union all
(select distinct customerid, last_updated as detection_date
            from db.my_table4
            where last_updated >= '"""+date+"""'
            and status = 'Enforced'
            and customerid::varchar in ("""+','.join(str(x) for x in cids)+"""));

select customerid, min(detection_date) as detection_date from base
group by 1;""")

print('overlapping:', check_duplicates.customer_id.nunique())

remove = check_duplicates['customerid'].unique()
selected_cids = selected_cids.loc[~selected_cids.customerid.isin(remove)]
selected_cids = selected_cids[['customerid']].drop_duplicates()
selected_cids.to_csv(LOCAL_DOWNLOAD_PATH+'auto_enforce_'+date+'.csv',sep=',',index=False,header=True)

#write out to EDX
sandbox_session = Session(session_folder='/home/ec2-user/SageMaker/')

# setup the edx data loader
edx_data_loader = sandbox_session.resource('EdxDataLoader')

if os.path.isfile(LOCAL_DOWNLOAD_PATH+'auto_enforce_'+date+'.csv'):
    edx_data_loader.upload_data_to_edx('myarn/["{}"]'.format(date), LOCAL_DOWNLOAD_PATH+'auto_enforce_'+date+'.csv')
else:
    pass
