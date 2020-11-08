import os, sys
import random
import socket
import torchvision.transforms as tf
import torch
from dgssr_trainer import model_fns, Trainer
from dl.data_loader.utils.get_p_l import get_p_l
from dl.dataset.ssr import SSRTest2, SSRTest
from dl.dataset.rotation import RotTest
from dl.data_loader.utils.ds2dl import train_DL_fn, test_DL_fn
import numpy as np
# /home/giorgio/Files/pycharm_project/pytorch_interpreter/bin/python3.7 /home/giorgio/Files/pycharm_project/CV/scripts/local/confusion_matrix.py
from openpyxl import Workbook
import math
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import matplotlib.pylab as plt

manualSeed = 0
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

class_name = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

wb = Workbook()
# grab the active worksheet
ws = wb.active

class Container:
    def __int__(self):
        pass


def confusion_matrix(model, data_loader, n_classes, ust_classes, device):
    # label_correct = 0
    n_correct = 0
    total = len(data_loader.dataset)
    # matrix = np.zeros((n_classes, n_classes))
    # matrix_ust = np.zeros((ust_classes, ust_classes))
    matrix_ust = [np.zeros((ust_classes, ust_classes)) for _ in range(n_classes)]

    for j, (norm_data, n, c_l, data) in enumerate(data_loader):
        norm_data, n, c_l = norm_data.to(device), n.to(device), c_l.to(device)
        n_logit, c_l_logit = model(norm_data)
        # _, c_l_pred = c_l_logit.max(dim=1)
        _, n_pred = n_logit.max(dim=1)
        # for (pred, real) in zip(c_l_pred, c_l):
        #     matrix[real][pred] += 1
        # for (pred, real) in zip(n_pred, n):
        #     matrix_ust[real][pred] += 1

        for (class_label, pred, real) in zip(c_l, n_pred, n):
            matrix_ust[class_label][real][pred] += 1

        # ## divide the resulting images into different folders
        # for i in range(len(n_pred)):
        #     temp = project_path + model_path + param + '/' + model_name+ \
        #         f'/{target}/{n[i].item()}_{n_pred[i]}/'
        #     if not os.path.exists(temp): os.makedirs(temp)
        #     image = tf.ToPILImage()(data[i])
        #     image.save(temp+f'{j}{i}'+'.jpg')

        ## total accuracy
        # label_correct += torch.sum(c_l_pred == c_l).item()
        n_correct += torch.sum(n_pred == n).item()
    # print(float(label_correct) / total, float(n_correct) / total)
    print(float(n_correct) / total, "all classes precision")

    # for i in range(matrix.shape[0]):
    #     matrix[i] = matrix[i]/sum(matrix[i])*100

    # for i in range(matrix_ust.shape[0]):
    #     matrix_ust[i] = matrix_ust[i] / sum(matrix_ust[i]) * 100
    for z in range(len(matrix_ust)):
        matrix_per_class = matrix_ust[z]
        n_correct_per_class = 0
        for i in range(matrix_per_class.shape[0]):
            n_correct_per_class += matrix_per_class[i][i]
        print(float(n_correct_per_class)/sum(sum(matrix_per_class)), class_name[z])


        for i in range(matrix_per_class.shape[0]):
            matrix_per_class[i] = matrix_per_class[i] / sum(matrix_per_class[i]) * 100

        # print(matrix_per_class)
        # print('纵轴为real，横轴为pre')
        print('********************************************')
        matrix_per_class = matrix_per_class.transpose()
    # matrix_ust = matrix_ust.transpose()
    # matrix = matrix.transpose()
    # return matrix, matrix_ust
    return matrix_ust, matrix_ust


def w(ws, i, j, value):
    ws[chr(ord('A') + j) + str(i + 1)] = value
def w_matrix(ws, begin_i, begin_j, matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            w(ws, i+begin_i, j+begin_j, matrix[i][j])

def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()

# 'photo', 'art_painting', 'cartoon' 'sketch'
if socket.gethostname() == "yujun":
    project_path = '/home/lyj/Files/project/pycharm/CV202009/'
else:
    project_path = '/media/autolab/1506ebe6-2e20-47c1-a0f6-9022bc6c122a/lyj/project/CV202009/'

model_path = 'data/cache/222server/1108_5/pacs_dg_r/caffenet/'
model_name = 'art_painting_cartoon'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_usv_classes = 4
model = model_fns['caffenet'](
    num_usv_classes=num_usv_classes,
    num_classes=-1)
model = model.to(device)
begin_i, begin_j = 0, 0
#for param in os.listdir(project_path+model_path):
for param in ['1.0_0.25']:
    output_file = f'{project_path}{model_path}{param}/{model_name}.result'
    print('redirect to ', output_file)
    # sys.stdout = open(output_file, 'w')
    print(f'model:{model_name}')
    # for target in ['photo', 'art_painting', 'cartoon', 'sketch']:
    for target in ['cartoon']:
        for prob in [0.25]:
            print('//////////////////////////////////////')
            # torch.cuda.set_device(0)
            model.load_state_dict(torch.load(project_path + model_path+param+f'/{model_name}.pkl', map_location=device))
            args = Container()
            args.image_size = 222
            test_paths, _, test_labels, _ = get_p_l(target, dir=project_path + 'data/test/')
            test_s_DS = SSRTest2(test_paths, test_labels, prob=prob, args=args)
            # test_s_DS = SSRTest(test_paths, test_labels, prob=prob, args=args)
            test_s_data_loader = test_DL_fn(test_s_DS, 128)

            # for i, (data, n, c_l) in enumerate(test_s_data_loader):
            #     data, n, c_l = data.to(device), n.to(device), c_l.to(device)
            model.eval()
            with torch.no_grad():

                # print(Trainer.test(model, test_s_data_loader, device))
                matrix, matrix_ust = confusion_matrix(model, test_s_data_loader, 7, num_usv_classes, device=device)

                ## write to excel
                #w(ws, begin_i, begin_j, param+target+str(prob))
                #begin_i += 1
                #w_matrix(ws, begin_i, begin_j, matrix)
                #w_matrix(ws, begin_i, begin_j+matrix.shape[0]+3, matrix_ust)
                #begin_i += matrix.shape[0]+3
                # for i in range(matrix.shape[0]):
                #     for j in range(matrix.shape[1]):
                #         w(ws, i, j, matrix[i][j])
                # for i in range(matrix_ust.shape[0]):
                #     for j in range(matrix_ust.shape[1]):
                #         w(ws, i, j+matrix.shape[0]+3, matrix[i][j])

                ## visulization
                # if prob == 1:
                #     #plt.figure(figsize=(16, 16))
                #     categories = ['dog', 'elep.', 'giraffe', 'guitar', 'horse', 'house', 'person']
                #     chart = sns.heatmap(matrix, annot=True, fmt='.1f', cmap='Reds', cbar=True,)
                #     chart.set_xticklabels(labels=categories)
                #     chart.set_yticklabels(labels=categories)
                #     plt.xlabel("Real")
                #     plt.ylabel("Predict")
                #     plt.title('Standard rotation')
                #     plt.show()
                # if prob == 0.25:
                #     #plt.figure(figsize=(16, 16))
                #     chart = sns.heatmap(matrix_ust, annot=True, fmt='.1f', cmap='Reds', cbar=True, )
                #     plt.xlabel("Real")
                #     plt.ylabel("Predict")
                #     plt.title('Standard rotation')
                #     plt.show()
                print(f'param:{param}')
                print(f'target domain:{target}')
                print(f'origin image prob:{prob}')
                print('-------------------------------------')
# wb.save("./confusion_m.xlsx")

