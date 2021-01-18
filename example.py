
# data = r'/data/t-shxiao/rec/data/new_alldocs_body.tsv'
data = r'C:\Users\stxiao\Desktop\scope_try\new_alldocs_body.tsv'
f = open('example_data/DocFeatures.tsv','w',encoding='utf-8')
cnt = 0
for line in open(data,'r',encoding='utf-8'):
    f.write(line)
    cnt+=1
    if cnt==500:
        break
f.close()
# from transformers import BertTokenizer
# a = BertTokenizer.from_pretrained('example_data/pretrainedModel/unilm2-base-uncased-vocab.txt',do_lower_case=True)
# print(a('fdafdfd'))


import torch


f = open('example_data/DocFeatures.tsv','r',encoding='utf-8')
cnt = 0
nids = []
for line in f:
    nid = line.strip().split('\t')[0]
    nids.append(nid)
    # cnt+=1
    # if cnt==100:
    #     break


import random
for i in range(8):
    f = open(f'example_data/testdata/ProtoBuf_{i}.tsv','w',encoding='utf-8')
    for j in range(500):

        hist = 'AA3i3hb;'
        h = random.sample(nids,1)[0]
        # print(h)
        hist = hist +';'+ h

        pos = 'BBUbdJN;'
        p = random.sample(nids,1)[0]
        pos = pos +';'+ p

        neg = 'BBUkAzR'
        n = random.sample(nids, 1)[0]
        neg = neg + ';' + n

        f.write(f'user_{100*i+j}'+'\t'+hist+'\t'+pos+'\t'+neg+'\n')


for i in range(8):
    f = open(f'example_data/traindata/ProtoBuf_{i}.tsv','w',encoding='utf-8')
    for j in range(500):
        s = ''
        for sess in range(random.randint(1,10)):
            s = s + 'BBUbdJN;'
            h = random.sample(nids, 1)[0]
            s += h
            s = s + '&'
            s = s + 'BBUasXo'
            n = random.sample(nids, 1)[0]
            s += n
            s = s + '|'
        s = s.strip('|')

        f.write(f'user_{100*i+j}'+'\t'+s+'\n')
#
