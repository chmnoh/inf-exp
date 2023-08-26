import os, requests, json, csv, sys, json

sentences = []
with open('tests/titles.lst','rb') as f:
    for i, line in enumerate(f):
        tt = line.strip().decode('utf8')
        sentences.append(tt)
        if i>=99: break

payload = {"sentences":sentences}
# print(payload)
headers = {'Content-Type': 'application/json'}
r = requests.post('http://localhost:64000/batchpredict', 
    data=json.dumps(payload), headers=headers)
# print(r.headers)
ret = json.loads(r.text)
# print(tt, ret)
for i, sent in enumerate(sentences):
    print(sentences[i], ret['classes'][i], ret['probs'][i])