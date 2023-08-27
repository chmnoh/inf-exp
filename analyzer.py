import os, sys, csv, sys, json, math, requests, kss

DEBUG = 0
def eval_tones(rec):
    sentences = []
    for i, line in enumerate(rec['bd']):
        for sent in kss.split_sentences(line):
            sentences.append(sent)
    payload = {"sentences": sentences}
    headers = {'Content-Type': 'application/json'}
    r = requests.post('http://localhost:64000/batchpredict',
            data=json.dumps(payload), headers=headers)
    ret = json.loads(r.text)
    rec['up'] = 0
    rec['down'] = 0
    for i, sent in enumerate(sentences):
        if DEBUG==1: print("{} => {}({})".format(sent, ret['classes'][i], ret['probs'][i]))
        if ret['classes'][i] == 'incr-future':
            rec['up'] += 1
        elif ret['classes'][i] == 'incr-present':
            rec['up'] += 0
        elif ret['classes'][i] == 'decr-future':
            rec['down'] += 1
        elif ret['classes'][i] == 'decr-present':
            rec['down'] += 0
        else:
            pass
    rec['s1'] = rec['up'] - rec['down']
    rec['s4'] = math.log(rec['up']+1) - math.log(rec['down']+1)

if __name__ == '__main__':
    dump_dir = "../models/news_dump/tab/0000"
    files = os.listdir(dump_dir)
    files.sort()
    tags = ["`pk'", "`dt'", "`ur'", "`tt'", "`bd'"]
    ktags = [t.strip("`'") for t in tags]
    wr = csv.writer(sys.stdout, delimiter=',', quotechar='"')
    for fn in files:
        if DEBUG==1: print("{}...".format(fn))
        with open("{}/{}".format(dump_dir, fn)) as fp:
            rec = {}
            tag_idx = 0
            k = ''
            for line in fp:
                if line.startswith(tags[tag_idx]):
                    if tag_idx == 0 and len(rec.keys()) > 0:
                        eval_tones(rec)
                        wr.writerow([rec['pk'], rec['dt'], rec['s1'], rec['s4'], 
                            rec['up'], rec['down'], rec['tt'], rec['ur']])
                        sys.stdout.flush()
                        rec = {}
                        if DEBUG == 1: break
                    k = ktags[tag_idx]
                    if k != 'bd':
                        rec[k] = line[len(tags[tag_idx]):].strip()
                    else:
                        rec[k] = [rec['tt']]
                        rec[k].append(line[len(tags[tag_idx]):].strip())
                    tag_idx += 1
                    if tag_idx == len(tags):
                        tag_idx = 0
                else:
                    if k == 'bd':
                        rec[k].append(line.strip())
                    else:
                        pass
            if len(rec.keys()) > 0:
                eval_tones(rec)
                wr.writerow([rec['pk'], rec['dt'], rec['s1'], rec['s4'], 
                    rec['up'], rec['down'], rec['tt'], rec['ur']])
                sys.stdout.flush()
                
