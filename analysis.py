import os, sys
import random
import socket
import torchvision.transforms as tf
import torch
from dgssr_trainer import model_fns, Trainer
from dl.data_loader.utils.get_p_l import get_p_l
from dl.dataset.ssr import SSRTest2, SSRTest
from dl.dataset.rotation import RotTest
from dl.dataset.tf_fn import norm_tf_fn
from dl.data_loader.utils.ds2dl import train_DL_fn, test_DL_fn
import numpy as np
# /home/giorgio/Files/pycharm_project/pytorch_interpreter/bin/python3.7 /home/giorgio/Files/pycharm_project/CV/scripts/local/confusion_matrix.py
from openpyxl import Workbook
import math
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pylab as plt
import pdb
from dl.utils.gradcam import visualize_cam, GradCAM, GradCAMpp, InputGrad


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


def means_logit_statistics(model, data_loader, n_classes, ust_classes, device):
    ust_means_logit = np.zeros((ust_classes, ust_classes))
    m = torch.nn.Softmax(dim=1)
    for j, (norm_data, n, c_l, data) in enumerate(data_loader):
        norm_data, n, c_l = norm_data.to(device), n.to(device), c_l.to(device)
        n_logit, c_l_logit = model(norm_data)
        n_logit = m(n_logit)
        for i in range(len(n)):

            for k in range(ust_classes):
                ust_means_logit[n[i]][k] += n_logit[i][k]
    for k in range(ust_classes):
        ust_means_logit[k] = ust_means_logit[k]/sum(ust_means_logit[k])

    print(ust_means_logit)
    # print(m(torch.Tensor(ust_means_logit)))


def weighted_mse_loss(t1, t2):
    return ((t1 - t2)**2*(t1 + t2)).means()


def attention_gradcam(model, data_loader, n_classes, ust_classes, device):
    layers = {}
    for name, m in model.named_modules():
        layers[name] = m
    gradcam = GradCAM(model, layers['features.conv5'])

    e = []
    ue = []

    for j, (norm_data, n, c_l, data) in enumerate(data_loader):
        norm_data, n, c_l = norm_data.to(device), n.to(device), c_l.to(device)
        data = data.to(device)


        # input_grad = InputGrad(model)
        n_logit, c_l_logit = model(norm_data)
        # _, c_l_pred = c_l_logit.max(dim=1)
        _, n_pred = n_logit.max(dim=1)


        # input = norm_data[k].unsqueeze(0)
        # logit = model(norm_data)[0]
        # gradients2 = input_grad.get_input_gradient(n_logit)



        for k in range(len(norm_data)):
            masks = torch.zeros(4, 222, 222)
            for rot_degree in range(4):

                rot_data = torch.rot90(data[k], rot_degree, [1, 2])
                norm_rot_data = norm_tf_fn(rot_data)
                mask, logit = gradcam(norm_rot_data, c_l[k])
                heatmap, cam_result = visualize_cam(mask, rot_data)
                mask = mask[0][0]
                average = mask.mean()
                # mask[mask<1.2*average]=0
                # mask[mask>=average]=1
                masks[rot_degree] = torch.rot90(mask, -rot_degree, [0, 1])
                print('logit', logit.max(dim=1)[1][0])
                if rot_degree != 0:
                    # print(rot_degree, c_l[k].item()==logit.max(dim=1)[1][0], torch.nn.MSELoss()(masks[0], masks[rot_degree]))
                    if c_l[k].item()==logit.max(dim=1)[1][0]:
                        # e.append(torch.nn.MSELoss()(masks[0], masks[rot_degree]))

                        e.append(weighted_mse_loss(masks[0], masks[rot_degree]))
                    else:
                        ue.append(torch.nn.MSELoss()(masks[0], masks[rot_degree]))

                # ## divide the resulting images into different folders
                # temp = project_path + model_path + param + '/' + model_name + \
                #        f'/{target}2/cam_real{c_l[k].item()}_pre{logit.max(dim=1)[1][0]}/'
                # if not os.path.exists(temp): os.makedirs(temp)
                # image = tf.ToPILImage()(cam_result)
                # image.save(temp + f'{j}_{k}_degree{rot_degree}' + '.jpg')






            # for x in range(ust_classes):
                ## test
                # activations = dict()
                # def forward_hook(module, input, output):
                #     activations['value'] = output
                #     activations['in'] = input
                #
                # layers['features.conv5'].register_forward_hook(forward_hook)
                # input = norm_data[k].unsqueeze(0)
                # logit = model(input)[0]
                # score = logit[range(len(logit)), logit.max(1)[-1]].sum()
                #
                # activations = activations['value']
                # gradients2 = torch.autograd.grad(score, activations, create_graph=True)[0]






                # mask, logit = gradcam(norm_data[k], x)
                # heatmap, cam_result = visualize_cam(mask, data[k])
                #
                # ## divide the resulting images into different folders
                # temp = project_path + model_path + param + '/' + model_name+ \
                #     f'/{target}_r/cam_{c_l[k].item()}_{n_pred[k]}/'
                # if not os.path.exists(temp): os.makedirs(temp)
                # image = tf.ToPILImage()(cam_result)
                # image.save(temp+f'{j}_{k}_{x}'+'.jpg')

            # tf.ToPILImage()(cam_result).show()
            # tf.ToPILImage()(masks[0]).show()
            # tf.ToPILImage()(masks[3]).show()
            # for i2 in range(4):
            #     tf.ToPILImage()(masks[i2]).show()
            # print()
        print(j)
        break
    print('e', len(e), np.array(e).mean())
    print('ue', len(ue), np.array(ue).mean())



def confusion_matrix(model, data_loader, n_classes, ust_classes, device):
    # label_correct = 0
    n_correct = 0
    total = len(data_loader.dataset)
    # matrix = np.zeros((n_classes, n_classes))
    # matrix_ust = np.zeros((ust_classes, ust_classes))
    matrix_ust = np.zeros((n_classes, ust_classes, ust_classes))

    for j, (norm_data, n, c_l, data) in enumerate(data_loader):
        norm_data, n, c_l = norm_data.to(device), n.to(device), c_l.to(device)
        n_logit, c_l_logit = model(norm_data)
        # _, c_l_pred = c_l_logit.max(dim=1)
        # pdb.set_trace()
        _, n_pred = n_logit.max(dim=1)
        # for (pred, real) in zip(c_l_pred, c_l):
        #     matrix[real][pred] += 1
        # for (pred, real) in zip(n_pred, n):
        #     matrix_ust[real][pred] += 1

        for (class_label, pred, real) in zip(c_l, n_pred, n):
            ## un supervised
            # matrix_ust[class_label][real][pred] += 1
            matrix_ust[0][class_label][pred] += 1

        ## divide the resulting images into different folders
        # for i in range(len(n_pred)):
        #     temp = project_path + model_path + param + '/' + model_name+ \
        #         f'/{target}/{n[i].item()}_{n_pred[i]}/'
        #     if not os.path.exists(temp): os.makedirs(temp)
        #     image = tf.ToPILImage()(data[i])
        #     image.save(temp+f'{j}{i}'+'.jpg')

        ## total accuracy
        # label_correct += torch.sum(c_l_pred == c_l).item()
        ## un supervised
        # n_correct += torch.sum(n_pred == n).item()
        ## supervised
        n_correct += torch.sum(n_pred == c_l).item()

    # print(float(label_correct) / total, float(n_correct) / total)
    print(float(n_correct) / total, "all classes precision")
    temp = matrix_ust.sum(axis=0)
    # print('**************************')
    # print(temp)
    for i in range(ust_classes):
        temp[i] /= sum(temp[i])
    # print(temp)
    # print('**************************')



    # for i in range(matrix.shape[0]):
    #     matrix[i] = matrix[i]/sum(matrix[i])*100

    # for i in range(matrix_ust.shape[0]):
    #     matrix_ust[i] = matrix_ust[i] / sum(matrix_ust[i]) * 100
    for z in range(len(matrix_ust)):
        matrix_per_class = matrix_ust[z]
        n_correct_per_class = 0
        for i in range(matrix_per_class.shape[0]):
            n_correct_per_class += matrix_per_class[i][i]
        # print(float(n_correct_per_class)/sum(sum(matrix_per_class)), class_name[z])


        for i in range(matrix_per_class.shape[0]):
            matrix_per_class[i] = matrix_per_class[i] / sum(matrix_per_class[i]) * 100

        # print(matrix_per_class)
        # print('纵轴为real，横轴为pre')
        # print('********************************************')


        # matrix_per_class = matrix_per_class.transpose()
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

## model path
# model_path = 'data/cache/222server/1110/pacs_dg_ssr_remove_local_rotation_info/caffenet/'
# model_path = 'data/cache/222server/patch_norm/pacs_dg_ssr_25_25/caffenet/'
# model_path = 'data/cache/222server/remove_all_norm/pacs_dg_ssr_25_25/caffenet/'
# model_path = 'data/cache/222server/1110/pacs_dg_ssr_remove_local_rotation_info/caffenet/'
# model_path = 'data/cache/222server/downstrem/dg_remove_local_rotation_info_pretrained/caffenet/'
# model_path = 'data/cache/222server/downstrem/dg_ssr_25_25_pretrained/caffenet/'
# model_path = 'data/cache/222server/downstrem/deep_all_r_invariant_dg_r_pretrained2/caffenet/'
model_path = 'data/cache/222server/downstrem/deep_all/caffenet/'
# model_path = 'data/cache/222server/remove_patch_norm/pacs_dg_r/caffenet/'

model_name = 'art_painting_cartoon'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = Container()
args.image_size = 222
# original_rotate lr_25_25 lr_25_5 lr_25_4 lr_25_3
# remove_patch_position_info remove_local_rotation_info deep_all
args.transformation_version = 'deep_all'
num_usv_classes = 7

model = model_fns['caffenet'](
    num_usv_classes=num_usv_classes,
    num_classes=-1)
model = model.to(device)
begin_i, begin_j = 0, 0
#for param in os.listdir(project_path+model_path):
for param in ['1.0_0.25']:
    ## redirect output
    # output_file = f'{project_path}{model_path}{param}/{model_name}.result'
    # print('redirect to ', output_file)
    # sys.stdout = open(output_file, 'w')

    print(f'model:{model_name}')
    # for target in ['art_painting', 'cartoon', 'sketch','photo']:
    for target in ['cartoon']:
        for prob in [0.25]:
            print(f'prob:{prob}//////////////////////////////////////')
            # torch.cuda.set_device(0)
            print('load', project_path + model_path+param+f'/{model_name}.pkl')
            model.load_state_dict(torch.load(project_path + model_path+param+f'/{model_name}.pkl', map_location=device))

            # for data_name in ['test/', 'train/', 'validate/']:
            for data_name in ['test/']:
                ## choose dataset
                test_paths, _, test_labels, _ = get_p_l(target, dir=project_path + 'data/'+data_name)
                test_s_DS = SSRTest2(test_paths, test_labels, prob=prob, args=args)
                # test_s_DS = SSRTest(test_paths, test_labels, prob=prob, args=args)
                test_s_data_loader = test_DL_fn(test_s_DS, 128)

                # for i, (data, n, c_l) in enumerate(test_s_data_loader):
                #     data, n, c_l = data.to(device), n.to(device), c_l.to(device)
                model.eval()
                attention_gradcam(model, test_s_data_loader, 7, num_usv_classes, device=device)

                with torch.no_grad():
                    pass
                    ## means logit statistics
                    # means_logit_statistics(model, test_s_data_loader, 7, num_usv_classes, device=device)
                #
                #     # print(Trainer.test(model, test_s_data_loader, device))
                #     matrix, matrix_ust = confusion_matrix(model, test_s_data_loader, 7, num_usv_classes, device=device)
                #
                    # ## write to excel
                    # w(ws, begin_i, begin_j, param+target+str(prob))
                    # for m in matrix_ust:
                    #     begin_i += 1
                    #     w_matrix(ws, begin_i, begin_j, m)
                    #     # w_matrix(ws, begin_i, begin_j+m.shape[0]+3, matrix_ust)
                    #     begin_i += m.shape[0]+1

                        ## supervised result
                        # for i in range(m.shape[0]):
                        #     for j in range(m.shape[1]):
                        #         w(ws, i, j, m[i][j])
                        # for i in range(matrix_ust.shape[0]):
                        #     for j in range(matrix_ust.shape[1]):
                        #         w(ws, i, j+matrix.shape[0]+3, matrix[i][j])
                #
                #     ## visulization
                #     # if prob == 1:
                #     #     #plt.figure(figsize=(16, 16))
                #     #     categories = ['dog', 'elep.', 'giraffe', 'guitar', 'horse', 'house', 'person']
                #     #     chart = sns.heatmap(matrix, annot=True, fmt='.1f', cmap='Reds', cbar=True,)
                #     #     chart.set_xticklabels(labels=categories)
                #     #     chart.set_yticklabels(labels=categories)
                #     #     plt.xlabel("Real")
                #     #     plt.ylabel("Predict")
                #     #     plt.title('Standard rotation')
                #     #     plt.show()
                #     # if prob == 0.25:
                #     #     #plt.figure(figsize=(16, 16))
                #     #     chart = sns.heatmap(matrix_ust, annot=True, fmt='.1f', cmap='Reds', cbar=True, )
                #     #     plt.xlabel("Real")
                #     #     plt.ylabel("Predict")
                #     #     plt.title('Standard rotation')
                #     #     plt.show()
                #     print(f'param:{param}')
                #     print(f'target domain:{target}')
                #     print(f'origin image prob:{prob}')
                #     print('-------------------------------------')
print(f'confusion_{project_path}{model_path}{param}/{model_name}.xlsx')
# wb.save(f'{project_path}{model_path}{param}/{model_name}_confusion.xlsx')

