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
from pickle import dump
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import tensorflow as tf
import tensorflow.keras as tk
import tensorflow_addons as tfa
from utils.py import reader, encode_cat, run_sql, load_in_feat_file_names, read_in_local_files, add_key


################################## Define variables #####################################################
LOCAL_DOWNLOAD_PATH = "home/guanxian/"
awsAccessKeyId = 'xyz'
awsSecretKey = '123'
bucket = 'moindex'


################################### Pull in numerical features raw data from S3 ################################

file_names = load_in_feat_file_names(awsAccessKeyId,awsSecretKey,bucket,'num_feat_training_')
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

num_feats = read_in_local_files('num_feat_training_')


########################################### Process numerical features ########################################

#convert date from string to datetime, add key (sessionid + date)
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

#derive numerical features list
num_feat = num_feats.columns.tolist()
for i in ['key','sessionid','date']:
    num_feat.remove(i)


############################################ Pull in categorical features raw data from S3 ##############################
file_names = load_in_feat_file_names(awsAccessKeyId,awsSecretKey,bucket,'cat_feat_training_')

#download in local dir
KEYS_TO_DOWNLOAD = file_names

def download_object(file_name):
    """Downloads an object from S3 to local."""

    download_path = LOCAL_DOWNLOAD_PATH + file_name[28:56]+'_'+file_name[67:70]+'.csv'

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
cat_feats = read_in_local_files('cat_feat_training_')
print('read in categorical feat, preprocess')


############################################ Process categorical features ########################################

#convert date, add key
cat_feats = add_key(cat_feats)

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

#derive categorical features list
cat_feat = cat_feats.columns.tolist()
for i in ['key','sessionid','date']:
    cat_feat.remove(i)

#encode cat feat with risk table
cat_feats[cat_feat] = cat_feats[cat_feat].astype(str)
cat_feats[cat_feat] = encode_cat(cat_feats[cat_feat], '/home/guanxian/categoricalVariableValsToRepresent.json','home/guanxian/categoricalVariableValsCountAndFraudRate.json')


#################################### Combine numerical and categorical features and save to final_feat.csv ###################################

all_feat = num_feat + cat_feat
with open(LOCAL_DOWNLOAD_PATH+'final_feat.csv','w') as f:
    for feat in all_feat:
        f.write(f"{feat}\n")


####################################### Pull in training data tags ##################################################
file_names = load_in_feat_file_names(awsAccessKeyId,awsSecretKey,bucket,'training_tags_')

#download in local dir
KEYS_TO_DOWNLOAD = file_names

def download_object(file_name):
    """Downloads an object from S3 to local."""

    download_path = LOCAL_DOWNLOAD_PATH + file_name[20:29]+'_'+file_name[48:50]+'.csv'

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

training_tags = read_in_local_files('training_tags_')

#add key
training_tags = add_key(training_tags)
################################ Merge numerical, categorical features and tags ###############################

dfs = [training_tags, num_feats, cat_feats]
training_final = ft.reduce(lambda left,right: pd.merge(left,right, on='key', how='inner'), dfs)


################################## Model training ##############################################
#split training data into train and test
x_train, x_test, y_train, y_test = train_test_split(training_final.drop(['tag','key','sessionid','date'],axis=1), training_final['tag'], test_size=0.25, random_state=111, shuffle=True)

#impute NA
x_train = x_train.fillna(-1)
x_test = x_test.fillna(-1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.copy())

x_test = scaler.transform(x_test)

#We first train an autoencoder with contrastive loss on the training set, we only need the encoder to get embedding layer
class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=0.2, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

def add_projection_head(encoder):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = keras.layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="encoder_with_projection_head"
    )
    return model

#pretrain the encoder
encoder = create_encoder()
encoder_with_projection_head = add_projection_head(encoder)
encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature),
)

encoder_with_projection_head.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0.002,
    patience=5,
    verbose=1,
    mode='min',
    restore_best_weights=True
)


cb = [early_stop]
history = encoder_with_projection_head.fit(x=x_train, y=y_train, batch_size=516, epochs=100,callbacks=cb)

#We then extract the embedding layer from the autoencoder, we expect the embedding layer including rich information to set apart good and risky sessions
x_train_k1 = encoder_with_projection_head.predict(x_train)
x_test_k1 = encoder_with_projection_head.predict(x_test)

#We then train a kmeans model on embedded train set samples which have tag = 1 (risky sessions)
#and apply the kmeans model to all embedded test set samples, to get
#1) a kmeans model,
#2) cluster membership of high purity risky clusters (defined as # of risky sessions / all sesions in a cluster >= 0.99)
#3) distance guardrail, which is max distance to high purity risky cluster' centroids
#We expect the high purity risk clusters' centroids from kmeans are like magnets that can attract risky sessions into the clusters
X_train_k1 = X_train_k1[y_train==1]
n_clusters = 20 ## number of clusters is determined by silhouette score
kmeans_model = MiniBatchKMeans(random_state=42, n_clusters = n_clusters)
kmeans_model.fit(X_train_k1)

df_embed_test = pd.DataFrame(X_test_k1)
df_embed_test['tag'] = y_test.reset_index(drop=True)

df_embed_test['cluster'] = kmeans_model.predict(X_test_k1)
df_embed_test['distances'] = np.min(kmeans_model.transform(X_test_k1),axis=1)

#check cluster purity and find high purity risky clusters
cluster_thr = {}
bad_cluster_list = []
for grp,df_grp in df_embed_test.groupby('cluster'): #grp is cluster, df_grp is all data belonging to that cluster
    cluster_purity = df_grp['tag'].sum()/((len(df_grp)-df_grp['tag'].sum())*6 + df_grp['BAA'].sum()) #class = 0 is downsampled by 1/6 in training data
    th = np.percentile(df_grp['distances'],cluster_purity*100) #corresponding dist to the cluster_purity (if cluster_purity/tag=1 rate is 50%, corresponding dist is median dist)
    cluster_thr[grp] = th
    if cluster_purity>0.99:
        bad_cluster_list.append(grp)
summary = pd.crosstab(df_embed_test['tag'],df_embed_test['cluster'].isin(bad_cluster_list))
print(summary)
distance_thr = df_embed_test[df_embed_test['cluster'].isin(bad_cluster_list)]['distances'].max() #longest distance in bad clusters
#so that by far we have kmeans model(fit on only risky sessions data), high purity risky clusters membership, distance threshold(max dist in bad clusters)

####################################### Save model artifacts ########################################

pkl.dump(scaler, open(LOCAL_DOWNLOAD_PATH + 'minmax_scaler','wb'))

json_model = encoder_with_projection_head.to_json()
json_file = open(LOCAL_DOWNLOAD_PATH+'ae_model.json', 'w')
json_file.write(json_model)

with open(LOCAL_DOWNLOAD_PATH+'ae_kmeans.pkl','wb') as f:
    pk.dump(kmeans_model, f)

with open(LOCAL_DOWNLOAD_PATH+'bad_clusters_.csv','w') as f:
    for c in bad_cluster_list:
        f.write(f"{c}\n")

with open(LOCAL_DOWNLOAD_PATH+'distance_thr_.csv','w') as f:
    for d in distance_thr:
        f.write(f"{d}\n")
