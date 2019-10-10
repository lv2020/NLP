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

    test_data=processor.get_test_data(args.data_dir)
    test_features=data2feature(test_data,args.max_seq_length,tokenizer)
    test_input_id=torch.tensor([f.input_ids for f in test_features],dtype=torch.long)
    test_input_mask=torch.tensor([f.input_mask for f in text_features],dtype=torch.long)
    test_label=torch.tensor([f.label for f in text_features],dtype=torch.long)

    test_data=TensorDataset(test_input_id,test_input_mask,test_label)
    test_sampler=SequentialSampler(test_data)
    test_dataloader=DataLoader(test_data,sampler=test_sampler,batch_size=args.eval_batch_size)

    model.eval()

    for input_ids,input_mask,label in test_dataloader:
        input_ids=input_ids.to(device)
        input_mask=input_mask.to(device)
        label=label.to(device)

        with torch.no_grad():
            logits = model(input_ids,input_mask)         
            pred = logits.max(1)[1]
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))

        logits = logits.detach().cpu().numpy()
        label = label.to('cpu').numpy()

    f1 = np.mean(metrics.f1_score(predict, gt, average=None))
    pre= np.mean(metrics.accuracy_score(predict,gt))
    recall=np.mean(metrics.recall_score(predict,gt))

    print('F1:%s \n pre:%s \n recall:%s'%(f1))

    return f1,pre,recall


 

            
        
    
