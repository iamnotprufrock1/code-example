#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from pandas import read_sql
import boto3
import datetime as dt
from datetime import date, datetime, timedelta
import os
import json
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from multiprocessing import Pool
from utils.py import get_f_n_r, read_categ_csv, categorical_combine, risk_table

LOCAL_DOWNLOAD_PATH = '/home/guanxian/'

#####################read in 1 month random sample data from production data to get risk table
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
dbname= 'mydb'
redshift_login = redshift_boto3.get_cluster_credentials( DbUser= session.owner_alias() + '_sais',
    DbName=dbname,
    ClusterIdentifier='mycluster',
    AutoCreate=True,
    DbGroups=['username'],
    DurationSeconds=3600)
conn = dbapi.connect(database=dbname,
                          host='myhost',
                          port=1234,
                          user=redshift_login['DbUser'],
                          password=redshift_login['DbPassword'],
                          ssl_context=True)
def run_sql(query, con = conn):
    df = read_sql(query, con= con)
    df.columns = [col for col in df.columns.values]
    return df
start_date = '2023-03-01'
end_date = '2023-03-31'
random_sample = run_sql("""with base as
(select sessionid, customerid,
       cat_feat1, cat_feat2, cat_feat3,
       cat_feat4, cat_feat5 from mytable1 where date between '"""+start_date+"""' and '"""+end_date+"""'
and challengetype = 'Pass'
and customerid is not null
and customerid <> 'NULL'
order by random() limit 1000000)
select distinct base.*
        ,case when baa.customerid is not null then 1 else 0 end as tag1
      ,case when rm.sessionid is not null then 1 else 0 end as tag2
      ,case when c.status in ('Fraud') then 1 else 0 end as tag3
       ,case when b.sessionid is not null then 1 else 0 end as tag4
        from base left join mytable2 c on c.customer_id::varchar = base.customerid
        left join mytable3 b on base.sessionid = b.sessionid
        and b.date between '"""+start_date+"""' and '"""+end_date+"""'
      left join mytable4 baa on base.customerid = baa.customer_id::varchar
      and date >= '"""+start_date+"""'
      left join mytable5 rm on rm.sessionid = base.sessionid
      and rm.date between '"""+start_date+"""' and '"""+end_date+"""';""")

print(random_sample.shape)
print(random_sample.sessionid.nunique())
print(random_sample.customerid.nunique())

#get a single tag col, tag = 1
random_sample.loc[(random_sample.tag1 == 1)|(random_sample.tag2 == 1)|(random_sample.tag3 == 1)|(random_sample.tag4 == 1),'tag'] = 1
random_sample.loc[random_sample.tag.isnull(),'tag'] = 0
random_sample.groupby(['tag']).agg({'sessionid': pd.Series.nunique,
                                   'customerid':pd.Series.nunique})

######################################train a risk table, add an 'unseen'
data = random_sample
fraud_col = 'tag'
cat = pd.read_csv(LOCAL_DOWNLOAD_PATH+'cat_feat_list.csv',sep=',')
cat = cat.columns.tolist()
cate_subdir = 'cat_vars'
alpha = random_sample.loc[random_sample.tag==1]['sessionid'].nunique()/random_sample['sessionid'].nunique()
print(alpha)

get_f_n_r(data, cat, data_dir=LOCAL_DOWNLOAD_PATH, cate_subdir = 'cat_vars', fraud_col = 'tag', alpha = alpha )

for var in cat:
    categorical_helper(var, high_card_freq_pct=2.0, minimum_freq= 5, top_n_freq= 20,
                      top_n_fraudrate=20, top_cumsum=0.8, max_n_distinct_val_cumsum=50,
                      cate_lst = cat)

categorical_combine(outputValsJSON = LOCAL_DOWNLOAD_PATH+'categoricalVariableValsToRepresent.json', multithread = 3,
                            minimum_freq=5, high_card_freq_pct= 2.0,
                            top_n_freq=20, top_n_fraudrate=20,
                            top_cumsum= 0.8, max_n_distinct_val_cumsum= 50,cate_lst = cat)

print('save risk table in dir', LOCAL_DOWNLOAD_PATH)
risk_table(outputRiskTableJSON=LOCAL_DOWNLOAD_PATH+'categoricalVariableValsCountAndFraudRate.json', multithread=20)
