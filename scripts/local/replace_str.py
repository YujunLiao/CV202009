import os

data_path = '/media/autolab/1506ebe6-2e20-47c1-a0f6-9022bc6c122a/lyj/project/CV202009/data/'
for mode in ['train/', 'validate/', 'test/']:
    for domain in os.listdir(data_path + mode):
        if os.path.isdir(data_path + mode + domain):
            continue

        lines = []
        for line in open(data_path + mode + domain):
            line = line.replace('/home/lyj/Files/project/pycharm/CV/data/PACS/kfold/', '/media/autolab/1506ebe6-2e20-47c1-a0f6-9022bc6c122a/lyj/data/pacs/')
            lines.append(line)
        file = open(data_path+mode+domain, 'w')
        for l in lines:
            file.write(l)





