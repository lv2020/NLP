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
from sklearn.model_selection import train_test_split
from torch import optim


from pytorch_pretrained_bert.optimization import BertAdam


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
        return self.get_data(os.path.join(data_dir, "train_balance.csv"),'train')
    
    def get_test_data(self,data_dir):
        return self.get_data(os.path.join(data_dir, "test.csv"),'test')
    
    def get_data(self,data_dir,types):

        datasets=[]
        f=open(data_dir,'r')
        ll=f.read().split('\n')
        for i in ll:
            if i=='':
              break
            text,label=i.split(',')
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
        input_ids = tokenizer.encode(example.text)
        input_mask=[1]*len(input_ids)
        if len(input_mask)>max_seq_length:
          input_ids=input_ids[:max_seq_length]
          input_mask=input_mask[:max_seq_length]
        while len(input_mask)<max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
        features.append(InputFeature(input_ids,input_mask,example.label))
        
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

def submit():
    f=pd.read_csv('./data/test_new.csv')
    test_text=[]
    text=[]
    tokenizer=BertTokenizer.from_pretrained('bert-base-chinese')
    for i in f['comment']:
        tokens=tokenizer.encode(i)
        text.append(i)
        if len(tokens)>max_seq_length:
            tokens=tokens[:max_seq_length]
        else:
            while len(tokens)<max_seq_length:
                tokens.append(0)
        test_text.append(tokens)
    test_data=TensorDataset(torch.tensor([i for i in test_text],dtype=torch.long))
    test_dataloader=DataLoader(test_data,batch_size=128)
    predict = np.zeros((0,), dtype=np.int32)
    res=[]
    for step,batch in enumerate(tqdm(test_dataloader,desc="iteration")):
        batch=tuple(t.to(device) for t in batch)
        input_ids=batch[0]
        with torch.no_grad():
            logits=model(input_ids)[0]
            pred=logits.max(1)[1]
            predict=np.hstack((predict,pred.cpu().numpy()))
    ids=[]
    for i in f['id']:
        ids.append(i)

    f=open('./sample.csv','w')
    f.write('id,label\n')
    for i in range(2000):
        f.write(ids[i]+','+str(predict[i])+'\n')
    f.close()

def main():
    max_seq_length=50
    f=open('./data/train_balance.csv','r')
    ll=f.read().split('\n')
    tokenizer=BertTokenizer.from_pretrained('bert-base-chinese')
    labels=[]
    texts=[]
    for i in ll:
        if i=='':
            break
        text,l=i.split(',')
        tokens=tokenizer.encode(text)
        if len(tokens)>max_seq_length:
            tokens=tokens[:max_seq_length]
        else:
            while len(tokens)<max_seq_length:
                tokens.append(0)
        texts.append(tokens)
        labels.append(int(l))

    train_data, dev_data,train_label, dev_label = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_dataset=TensorDataset(torch.tensor([i for i in train_data],dtype=torch.long),torch.tensor([i for i in train_label],dtype=torch.long))
    dev_dataset=TensorDataset(torch.tensor([i for i in dev_data],dtype=torch.long),torch.tensor([i for i in dev_label],dtype=torch.long))
    train_dataloader=DataLoader(train_dataset,batch_size=128)
    dev_dataloader=DataLoader(dev_dataset,batch_size=128)

    model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for _ in trange(10,desc="epochs"):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader,desc="iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids,label = batch
            output = model(input_ids,labels=label)
            loss=output[0]

            loss.backward()
            optimizer.step()
            model.zero_grad()
        print('loss= %s'%(loss))

        model.eval()
        predict = np.zeros((0,), dtype=np.int32)
        gt = np.zeros((0,), dtype=np.int32)
        for step, batch in enumerate(tqdm(dev_dataloader,desc="iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids,label = batch

            with torch.no_grad():
                logits = model(input_ids)[0]       
                pred = logits.max(1)[1]
                predict = np.hstack((predict, pred.cpu().numpy()))
                gt = np.hstack((gt, label.cpu().numpy()))

            logits = logits.detach().cpu().numpy()
            label = label.to('cpu').numpy()
        f1 = np.mean(metrics.f1_score(predict, gt, average=None))
        pre= np.mean(metrics.accuracy_score(predict,gt))
        recall=np.mean(metrics.recall_score(predict,gt))

        print('F1:%s \n pre:%s \n recall:%s'%(f1,pre,recall))




 
    
