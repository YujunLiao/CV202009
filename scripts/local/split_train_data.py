import os

path = '/home/lyj/Files/project/pycharm/CV202009/data/'
dataset = 'PACS/'
for mode in ['train/']:
    for domain in os.listdir(path + mode + dataset):
        if os.path.isdir(path + mode + dataset + domain):
            continue

        validate_data = open(path + 'validate/' + domain, 'w')
        train_data = open(path + 'train/' + domain, 'w')

        lines_per_class = [[] for i in range(7)]
        for line in open(path + mode + dataset + domain):
            class_label = int(line.split(" ")[1][:-1])
            lines_per_class[class_label].append(line)
        for i in range(7):
            total_n = len(lines_per_class[i])
            for l in lines_per_class[i][:int(0.9*total_n)]:
                train_data.writelines(l)
            for l in lines_per_class[i][int(0.9*total_n):]:
                validate_data.writelines(l)




