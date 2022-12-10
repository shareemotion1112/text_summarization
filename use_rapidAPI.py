import pandas as pd
import os
import requests
import re
import requests
import json



url = "https://tldrthis.p.rapidapi.com/v1/model/abstractive/summarize-text/"

HEADER = {
	"content-type": "application/json",
	"X-RapidAPI-Key": "1ad8db7eeamsh68f565e6c17874cp150656jsn2f7238e94c76",
	"X-RapidAPI-Host": "tldrthis.p.rapidapi.com"
}


def request_summarize(payload_txt):

    payload = {
    	"text": payload_txt,
    	"min_length": 100,
    	"max_length": 300
    }    
    response = requests.request("POST", url, json=payload, headers=HEADER)
    
    return (response.text)


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
        
        summarized_txt = request_summarize(payload_txt)
        
        chapter_sum.append(summarized_txt)
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

    
