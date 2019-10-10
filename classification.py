#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:22:35 2019

@author: lvvv
"""


import os
import codecs
import random
import logging
import argparse
from tqdm import tqdm, trange

from sklearn import metrics
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

class InputData():
    def __init__(self,text,label):
        self.text=text
        self.label=label

class InputFeature():
    def __init__(self,input_ids,input_mask,label):
        self.input_ids=input_ids
        self.input_mask=input_mask
        self.label=label
    
class DataProcessor():
    '''
    Args:
        data_dir: 训练文件路径
    Return:
        datasets: 训练数据集，包括文本和标签两部分
    '''
    def get_train_data(self,data_dir):
        return self.get_data(self,os.path.join(data_dir, "train.csv"),'train')
    
    def get_test_data(self,data_dir):
        return self.get_data(self.os.path.join(data_dir, "test.csv"),'test')
    
    def get_data(self,data_dir,types):
        logger.info("loading %s data" % (types))
        datasets=[]
        f=pd.read_csv(data_dir)
        for i in f['label\tcomment']:
            label,text=i.split('\t')
            datasets.append(InputData(text,int(label)))
        return datasets


def data2feature(datasets,max_seq_length,tokenizer):
    '''
    word2vec
    Args:
        datasets        :   输入数据集，由DataProcessor导出
        max_seq_length  :   最大文本长度
        tokenizer       :   分词方法
    Return:
        features:
            input_ids   :   每个词的ID，对应一个向量
            input_mask  :   真实字符对应0，补全字符对应1
            label       :   标签
        
    '''
    features=[]
    for example in datasets:
        token = tokenizer.tokenize(example.text)
        
        '''
        为[CLS]和[SEP]留下位置
        '''
        if len(token) > max_seq_length - 2:
            token = token[0:(max_seq_length - 2)]
            
        tokens=[]
        tokens.append('[CLS]')
        for i in token:
            tokens.append(i)
        tokens.append("[SEP]")
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 for i in input_ids]
        
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            
        assert len(input_ids)==max_seq_length
        assert len(input_mask)==max_seq_length
        
        features.append(InputFeature(input_ids,input_mask,label))
        
    return features

def test(model,processor,args,tokenizer,device):
    '''
    测试集
    Args:
        model       :模型
        processor   :数据预处理方法
        args        :参数列表
        tokenizer   :分词方法
        device      :是否使用GPU
    Return:
        f1          :F1
        pre         :准确率
        recall      :召回率
    '''
    
            
        
    
