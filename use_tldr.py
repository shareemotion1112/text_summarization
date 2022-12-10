import pandas as pd
import re
import json
import torch
from fairseq.models.bart import BARTModel
from tqdm import tqdm
import os
from os.path import join
import logging
from time import time
import numpy as np


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



""" 모델 생성 """
NAME = "SciTLDR-FullText"
pp = path + f"\\SciTLDR-Data\\{NAME}-bin"
checkpoint_file = path + '\\pretrained_model\\scitldr_bart-xsum.tldr-aic.pt'
checkpoint_dir = path + '\\checkpoints'


bart = BARTModel.from_pretrained(checkpoint_dir, checkpoint_file=checkpoint_file, data_name_or_path = pp, task='translation')

if torch.cuda.is_available():
    bart.cuda()
    bart.half()

bart.eval()

# hyper-parameter
beam = 2 #2
lenpen = 0.4 #0.4
no_repeat_ngram_size = 3 #3
max_len_b = 1000 #50
min_len = 10 #5
STEP = 1000


datadir = path + f"\\SciTLDR-Data\\{NAME}"
source_fname = join(datadir, 'test.source')
pred_fname = join(path + '\\outputs', str(np.ceil(time())))

def get_summarization(payload):
    hypotheses_batch = bart.sample(payload, beam=beam, 
                                    lenpen=lenpen, 
                                    max_len_b=max_len_b,
                                    min_len=min_len,
                                    no_repeat_ngram_size=no_repeat_ngram_size)
    print(hypotheses_batch)
    return hypotheses_batch



""" 실행 구문 """
chapter_summary = {}

for i in range(1, len(chapter_index)):
    chapter_sum = []    
    
    n = 1
    flag = True
    while flag:
        ind_start = chapter_index[i-1] + 1 + (n-1) * STEP
        ind_end = ind_start + n * STEP
        # print(ind_start, ind_end)
        if ind_end > chapter_index[i]:
            ind_end = chapter_index[i]
            flag = False
        list_chapter_txt = book[ind_start:ind_end]
        
        payload_txt = ' '.join(list_chapter_txt)       

        payload_txt = re.sub('\\n', ' ', payload_txt)

        chapter_sum = get_summarization(payload_txt)
        
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