import os, json, random

dirs = ['0-incr-future',
    '1-incr-present',
    '2-neut', 
    '3-decr-future',  
    '4-decr-present'
]
for d in dirs:
    txt_files = os.listdir("{}/data".format(d))
    txt_files.sort()
    all_texts = []
    for f in txt_files:
        fpath = "{}/data/{}".format(d,f)
        with open(fpath, 'rb') as rf:
            for line in rf.read().decode('utf-8').split('\n'):
                if len(line.strip('\n')) == 0: continue
                all_texts.append(line[line.find(' ')+1:].strip('"'))
    label = d[0]
    ratio = {"train":0.9, "test":0.1}
    offset = 0
    random.shuffle(all_texts)
    total = len(all_texts)
    for i,x in enumerate(['train','test']):
        outd = "{}/{}".format(x,label)
        if not os.path.exists(outd):
            os.makedirs(outd, exist_ok=True)
        num = int(total*ratio[x])
        if i==len(ratio.keys())-1:
            num = total-offset
        with open("{}/sentences.jsonl".format(outd), "w+") as wf:
            for line in all_texts[offset:offset+num]:
                json_line = json.dumps({"text":line})
                wf.write(json_line+'\n')
        offset += num
