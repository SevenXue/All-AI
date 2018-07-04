# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from time import time
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report




def to_numeric(df, columns={'int':['grade','bid_num','bid_order_num','bid_goods_num'],'float':['price','bid_money_sum']}):

    for i in columns['int']:
        df[i] = df[i].astype(int)

    for i in columns['float']:
        df[i] = df[i].astype(float)
    return df

def positives_generator(raw_file_path):
    a = time()
    raw = pd.read_csv(raw_file_path, header=None, sep='\t', na_filter=True, keep_default_na=False,
                      na_values=['NULL', 'NA'],
                      dtype={'shop_id': np.int32, 'buyer_nick': str, 'buyer_id': float, 'num_iid': np.int32,
                                 'item_id': int,
                                 'cid': int, 'price': float, 'item_feature': str, 'buyer_feature': str,
                                 'source_time': int,
                                 'recent_pay_time': int, 'bid_num': int, 'bid_order_num': int, 'bid_goods_num': int,
                                 'buy_money_sum': float,
                                 'grade': int, 'interval': int})
    raw.columns = ['shop_id', 'buyer_nick', 'buyer_id', 'num_iid', 'item_id', 'cid', 'price', 'item_feature',
                       'buyer_feature', 'source_time', 'recent_pay_time', 'bid_num', 'bid_order_num', 'bid_goods_num',
                       'bid_money_sum', 'grade', 'interval']

    # 缺失值处理
    raw = raw.replace('\N', np.nan)
    raw.dropna(inplace=True)

    raw[['if1', 'if2', 'if3', 'if4', 'if5', 'if6', 'if7', 'if8', 'if9', 'if10']] = raw['item_feature'].str[1:-1].str.split(',', expand=True).astype(float)
    raw[['bf1', 'bf2', 'bf3', 'bf4', 'bf5', 'bf6', 'bf7', 'bf8', 'bf9', 'bf10']] = raw['buyer_feature'].str[1:-1].str.split(',', expand=True).astype(float)

    raw = raw[
        ['price',
         'if1', 'if2', 'if3', 'if4', 'if5', 'if6', 'if7', 'if8', 'if9', 'if10',
         'bf1', 'bf2', 'bf3', 'bf4', 'bf5', 'bf6', 'bf7', 'bf8', 'bf9', 'bf10',
         'bid_num', 'bid_order_num', 'bid_goods_num', 'bid_money_sum', 'grade']].loc[raw['interval']>86400*3]

    raw['interest'] = raw['if1'] * raw['bf1'] + raw['if2'] * raw['bf2'] + raw['if3'] * raw['bf3'] + raw['if4'] * raw[
        'bf4'] + raw['if5'] * raw['bf5'] + raw['if6'] * raw['bf6'] + raw['if6'] * raw['bf6'] + raw['if7'] * raw['bf7'] + \
                      raw['if8'] * raw['bf8'] + raw['if9'] * raw['bf9'] + raw['if10'] * raw['bf10']

    raw['label'] = 1

    # 特征处理
    raw = to_numeric(raw)

    raw = raw.sample(frac=1.0).reset_index(drop=True)

    t = time() - a
    print('positives finished with {} data, cost {}s'.format(raw.shape[0], t))

    return raw

def negatives_generator(raw_file_path):

    a = time()
    raw = pd.read_csv(raw_file_path, header=None, sep='\t', na_filter=True, keep_default_na=False,
                    na_values=['NULL', 'NA'],
                          dtype={'shop_id': np.int32, 'buyer_nick': str, 'num_iid': np.int32,
                                 'item_id': int, 'price': float, 'item_feature': str, 'buyer_feature': str,
                                 'source_time': int, 'recent_pay_time': int, 'bid_num': int,
                                 'bid_order_num': int, 'bid_goods_num': int, 'buy_money_sum': float, 'grade': int})
    raw.columns = ['shop_id', 'buyer_nick', 'num_iid', 'item_id', 'price', 'item_feature',
                       'buyer_feature', 'source_time', 'recent_pay_time', 'bid_num', 'bid_order_num', 'bid_goods_num', 'bid_money_sum','grade']

        # 缺失值处理
    raw = raw.sample(frac=0.1)
    raw = raw.replace('\N', np.nan)
    raw.dropna(inplace=True)

    raw[['if1', 'if2', 'if3', 'if4', 'if5', 'if6', 'if7', 'if8', 'if9', 'if10']] = raw['item_feature'].str[1:-1].str.split(',',expand=True).astype(float)
    raw[['bf1', 'bf2', 'bf3', 'bf4', 'bf5', 'bf6', 'bf7', 'bf8', 'bf9', 'bf10']] = raw['buyer_feature'].str[1:-1].str.split(',', expand=True).astype(float)

    raw = raw[
            ['price',
             'if1', 'if2', 'if3', 'if4', 'if5', 'if6', 'if7', 'if8', 'if9', 'if10',
             'bf1', 'bf2', 'bf3', 'bf4', 'bf5', 'bf6', 'bf7', 'bf8', 'bf9', 'bf10',
             'bid_num', 'bid_order_num', 'bid_goods_num', 'bid_money_sum', 'grade']]

    raw['interest'] = raw['if1']*raw['bf1']+raw['if2']*raw['bf2']+raw['if3']*raw['bf3']+raw['if4']*raw['bf4']+raw['if5']*raw['bf5']+raw['if6']*raw['bf6']+raw['if6']*raw['bf6']+raw['if7']*raw['bf7']+raw['if8']*raw['bf8']+raw['if9']*raw['bf9']+raw['if10']*raw['bf10']

    raw['label'] = 0

        #时间特征处理
    raw = to_numeric(raw)

    raw = raw.sample(frac=1).reset_index(drop=True)

    t = time() - a
    print('negatives finished with {} data, cost {}s'.format(raw.shape[0], t))

    return raw


#生成正样本
positives = positives_generator('rebuy_all.txt')
print positives[:5]
       
#生成负样本
negatives = negatives_generator('not_rebuy_all.txt')
print negatives[:5]

#拆分训练集和测试集
pos_train_num = int(positives.shape[0]*0.8)
neg_train_num = int(negatives.shape[0]*0.8)
        
train = positives[:pos_train_num].append(negatives[:neg_train_num], ignore_index=True)
train = train.sample(frac=1).reset_index(drop=True)
x_train = train[[i for i in train.columns if i!='label']]
y_train = train['label']

test = positives[pos_train_num:].append(negatives[neg_train_num:], ignore_index=True)
test = test.sample(frac=1).reset_index(drop=True)
x_test = test[[i for i in train.columns if i!='label']]
y_test = test['label']


#开始调参

gbdt = GradientBoostingClassifier(
    init=None,
    learning_rate=0.1,
    loss='deviance',
    n_estimators=1000,
    subsample=0.8,
    verbose=0,
    max_leaf_nodes=30,
)
gbdt.fit(x_train, y_train)
y_pred = gbdt.predict(x_test)

print gbdt.feature_importances_

print classification_report(y_test, y_pred)
