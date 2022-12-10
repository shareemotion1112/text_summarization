import pandas as pd
import re
import json
import torch
from fairseq.models.bart import BARTModel
from tqdm import tqdm
import os
from os.path import join
import logging
import time



path = os.path.dirname(os.path.realpath('__file__'))
os.chdir(path)

txt = pd.read_csv('ulysses.txt', engine='python', sep=',', quotechar='"', error_bad_lines=False)

with open('ulysses.txt', 'rt', encoding='UTF-8') as f:
    line = f.readlines()

    
book_org = line[71:]
book = [v for v in book_org if v != '\n'] # 한줄 띄우기 제거 


reg = re.compile(r'\[ ([0-9])\ ]')

chapter_index = []
i = 0
for line in book:
    if reg.match(line):
        print(i, line)
        chapter_index.append(i)
    i += 1


""" 임시 """
for i in range(1, len(chapter_index)):
    chapter_sum = []    
    
    n = 1
    flag = True
    while flag:
        ind_start = chapter_index[i-1] + 1 + (n-1) * 100
        ind_end = ind_start + n * 100
        # print(ind_start, ind_end)
        if ind_end > chapter_index[i]:
            ind_end = chapter_index[i]
            flag = False
        list_chapter_txt = book[ind_start:ind_end]
        
        payload_txt = ' '.join(list_chapter_txt)       
        
        print(f"{ind_start} -- {ind_end}")
        n += 1


""" 모델 생성 """
checkpoint_dir = path + '\\pretrained_model'
os.chdir(checkpoint_dir)
model_file = 'scitldr_bart-xsum.tldr-aic.pt'
# checkpoint_dir_replaced = re.sub("\\\\", '//', checkpoint_dir)
with open('payload.txt', 'wt', encoding='UTF-8') as f:
    f.writelines(payload_txt)


bart = BARTModel.from_pretrained(checkpoint_dir, checkpoint_file=model_file, task='translation')












""" 실행 구문 """
chapter_summary = {}

for i in range(1, len(chapter_index)):
    chapter_sum = []    
    
    n = 1
    flag = True
    while flag:
        ind_start = chapter_index[i-1] + 1 + (n-1) * 100
        ind_end = ind_start + n * 100
        # print(ind_start, ind_end)
        if ind_end > chapter_index[i]:
            ind_end = chapter_index[i]
            flag = False
        list_chapter_txt = book[ind_start:ind_end]
        
        payload_txt = ' '.join(list_chapter_txt)       
        
        print(f"{ind_start} -- {ind_end}")
        n += 1
        
        
    chapter_summary[i] = chapter_sum

result = {}
for key in chapter_summary.keys():
    tmp = chapter_summary[key]       
    summary = ''
    for tt in tmp:
        summary += tt[13:(len(tt) - 2)]
    
    result[key] = summary



# Serialize data into file:
json.dump( result, open( "tldr_summary.json", 'w' ) )

# Read data from file:
data = json.load( open( "tldr_summary.json" ) )