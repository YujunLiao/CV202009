import random
import numpy as np
import copy
import os

import sys
from os.path import dirname
from time import time, strftime, localtime
import socket
from math import ceil
import torch
from torch import optim
from torch import nn
import argparse
import wandb
from dl.data_loader.dgr import get_DGR_data_loader
from dl.data_loader.dgssr import get_DGSSR_data_loader
from dl.optimizer import get_optimizer
from dl.utils.gradcam import InputGrad, visualize_cam
from dl.utils.writer import Writer
from dl.utils.s2t import ms2st, ss2st
from dl.utils.pp import pretty_print as pp
from dl.model import caffenet, resnet, mnist
from dl.utils.result import Result
from torchvision import transforms as tf
from dl.utils.mm import monitor_memory

manualSeed = 0
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)




model_fns = {
    'caffenet': caffenet.caffenet,
    'resnet18': resnet.resnet18,
    # 'alexnet': alexnet.alexnet,
    'resnet50': resnet.resnet50,
    'lenet': mnist.lenet
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--network", default='caffenet')
    parser.add_argument("--source", nargs='+')
    parser.add_argument("--target")
    parser.add_argument("--num_classes", "-c", type=int, default=7)
    parser.add_argument("--num_usv_classes", type=int, default=4)
    parser.add_argument('--transformation_version', type=str,
                        default='deep_all')

    parser.add_argument("--domains", nargs='+',
                        default=['sketch'])
    parser.add_argument("--targets", nargs='+', default=['photo'])
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--parameters", nargs='+', default=[[0.7, 0.7],[0.5, 0.75]],
                        type=lambda params:[float(_) for _ in params.split(',')])


    parser.add_argument("--usvt_weight", type=float)
    parser.add_argument("--original_img_prob", default=None, type=float)
    parser.add_argument("--epochs", "-e", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=99)
    parser.add_argument("--learning_rate", "-l", type=float, default=.01)
    parser.add_argument("--image_size", type=int, default=222)
    parser.add_argument("--val_size", type=float, default="0.1")
    parser.add_argument("--collect_per_batch", type=int, default=5)

    # parser.add_argument("--tf_logger", type=bool, default=True)
    # parser.add_argument("--folder_name", default=None)
    parser.add_argument("--data_dir",
                        default=f'{dirname(__file__)}/data/')
    parser.add_argument("--output_dir", default=f'{dirname(__file__)}/output/')
    parser.add_argument("--redirect_to_file", default='null')
    parser.add_argument("--experiment", default='DG_ssr_PACS')
    parser.add_argument("--load_model", default=None)
    parser.add_argument("--wandb", default=False, action='store_true')

    parser.add_argument("--classify_only_original_img", type=bool, default=True)
    parser.add_argument("--max_num_s_img", default=100, type=int)
    parser.add_argument("--max_num_t_img", default=-1, type=int)
    parser.add_argument("--train_all_param", default=True, type=bool)
    parser.add_argument("--nesterov", default=False, action='store_true')

    parser.add_argument("--TTA", default=False, action='store_true')
    parser.add_argument("--min_scale", default=0.8, type=float)
    parser.add_argument("--max_scale", default=1.0, type=float)
    parser.add_argument("--random_horiz_flip", default=0.0, type=float)
    parser.add_argument("--jitter", default=0.0, type=float)
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float)

    return parser.parse_args()


class Trainer:
    def __init__(self, args, model, data_loaders, optimizer, scheduler, writer):

        # self.args = args.args
        self.args = args
        self.device = args.device
        self.model = model.to(args.device)
        self.input_grad = InputGrad(model)
        model.input_grad = self.input_grad
        self.writer = writer
        self.train_data_loader, \
        self.val_s_data_loader, self.val_us_data_loader, \
        self.test_s_data_loader, self.test_us_data_loader = data_loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.test_s_loaders = {"val_s": self.val_s_data_loader,
                             "test_s": self.test_s_data_loader}
        self.test_us_loaders = {"val_us": self.val_us_data_loader,
                               "test_us": self.test_us_data_loader}

        self.cur_epoch = -1
        # collect_frequency
        self.collect_per_batch = self.args.collect_per_batch
        self.results = {"val_s": torch.zeros(self.args.epochs),
                        "val_us": torch.zeros(self.args.epochs),
                        "test_s": torch.zeros(self.args.epochs),
                        "test_us": torch.zeros(self.args.epochs)}

        self.train()

    def train(self):
        start_time = time()
        pp('Start training')
        pp({'train':self.train_data_loader.dataset,
            'validation': self.val_s_data_loader.dataset,
            'test': self.test_s_data_loader.dataset})
        pp(vars(self.args))

        # TODO(lyj):
        for self.cur_epoch in range(self.args.epochs):
            self.train_epoch()


        r = Result()
        r.v_s_b_i = self.results["val_s"].argmax()
        r.t_s_b_i = self.results["test_s"].argmax()
        r.val_s_best = self.results["val_s"].max()
        r.test_s_best = self.results["test_s"].max()
        r.test_s_select = self.results["test_s"][r.v_s_b_i]
        r.u_when_s = self.results["test_us"][r.v_s_b_i]

        r.v_u_b_i = self.results["val_us"].argmax()
        r.t_u_b_i = self.results["test_us"].argmax()
        r.val_u_best = self.results["val_us"].max()
        r.test_u_best = self.results["test_us"].max()
        r.test_u_select = self.results["test_us"][r.v_u_b_i]
        r.s_when_u = self.results["test_s"][r.v_u_b_i]

        r.v_s_means = self.results["val_s"].mean()
        r.t_s_means = self.results["test_s"].mean()
        r.v_u_means = self.results["val_us"].mean()
        r.t_u_means = self.results["test_us"].mean()

        r.v_s_std = self.results["val_s"].std()
        r.t_s_std = self.results["test_s"].std()
        r.v_u_std = self.results["val_us"].std()
        r.t_u_std = self.results["test_us"].std()


        # print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        temp_dict = {
            'now': strftime("%Y-%m-%d %H:%M:%S", localtime()),
            'source': self.args.source,
            'target':self.args.target,
            'param':self.args.params,
            'bs': self.args.batch_size,
            'lr': self.args.learning_rate,
            # 'Highest accuracy on validation set appears on epoch': r.v_s_b_i.item(),
            'Highest accuracy on validation set appears on epoch': r.v_u_b_i.item(),
            'Highest accuracy on test set appears on epoch': r.t_u_b_i.item(),
            'Accuracy on test set when the accuracy on validation set is highest': r.test_u_select.item(),
            'Highest accuracy on test set': r.test_u_best.item(),
            'duration': time() - start_time
        }
        pp(temp_dict)
        pp(vars(r))
        self.writer.w(temp_dict)
        if self.args.wandb:
            # wandb.log({'r/test_select': test_s_select.item(),
            #             'r/val_best': val_s_best.item(),
            #            'r/test_best': test_s_best.item(),
            #            'r/v_b_i': v_s_b_i,
            #            'r/t_bi': t_s_b_i})
            wandb.log(vars(r))
            # table = wandb.Table(columns=[
            #     f'{self.args.source[0]}->{self.args.target}-{"-".join([str(_) for _ in self.args.params])}',
            #     "val_best", "test_best", "test_select"])
            # table.add_data("epoch", v_b_i, t_b_i, "#")
            # table.add_data("acc", val_best.item(), test_best.item(), test_select.item())
            # wandb.log({"summary": table})

        torch.cuda.empty_cache()



    def train_epoch(self):
        print('******************************')
        self.scheduler.step()
        lrs = self.scheduler.get_lr()
        criterion = nn.CrossEntropyLoss()

        # Set the mode of the model to trainer, then the parameters can begin to be trained

        for i, (data, n, c_l) in enumerate(self.train_data_loader):
            self.model.train()
            data, n, c_l = data.to(self.device), n.to(self.device), c_l.to(self.device)
            # for i in range(20):
            #     print(n[i])
            #     tf.ToPILImage()(data[i]).show()
            #     print()
            self.optimizer.zero_grad()

            n_logit, c_l_logit = self.model(data)  # , lambda_val=lambda_val)
            # us_loss = criterion(n_logit, n)



            ## temporary used
            c_l_logit = n_logit
            # c_l_logit = n_logit+n_logit_2
            # s_loss = criterion(c_l_logit[n == 0], c_l[n == 0])
            s_loss = criterion(c_l_logit, c_l)
            # s_loss = torch.tensor(-1)


            _, c_l_pred = c_l_logit.max(dim=1)
            # _, n_pred = n_logit.max(dim=1)
            # _, domain_pred = domain_logit.max(dim=1)


            # # self.model.eval()
            # n_random = torch.randint(0, 4, (len(data), ))
            #
            # n_logit2, _ = self.model(data)
            # activations = self.input_grad.activations['out']
            #
            # grad = self.input_grad.get_input_gradient(n_logit2, create_graph=True)
            # abs_grad = torch.norm(grad)
            #
            # mask = self.input_grad.get_mask(activations, grad).data
            # # k = 5
            # # tf.ToPILImage()(visualize_cam(mask[k], data[k])[0]).show()
            # # tf.ToPILImage()(visualize_cam(mask[k], data[k])[1]).show()
            #
            # grad *= 100000
            #
            #
            # data_r = torch.zeros_like(data)
            # for j in range(len(data)): data_r[j] = torch.rot90(data[j], -n_random[j].item(), [1, 2])
            # n_logit_r, _ = self.model(data_r)
            # activations_r = self.input_grad.activations['out']
            #
            #
            # grad_r = self.input_grad.get_input_gradient(n_logit_r, create_graph=True)
            # abs_grad_r = torch.norm(grad_r)
            #
            # mask_r = self.input_grad.get_mask(activations_r, grad_r)
            # # tf.ToPILImage()(visualize_cam(mask_r[k], data_r[k])[0]).show()
            # # tf.ToPILImage()(visualize_cam(mask_r[k], data_r[k])[1]).show()
            #
            # for j in range(len(mask_r)): mask_r[j] = torch.rot90(mask_r[j], n_random[j].item(), [1, 2])
            #
            # grad_r_r = torch.zeros_like(grad_r)
            # for j in range(len(grad_r)): grad_r_r[j] = torch.rot90(grad_r[j], n_random[j].item(), [1, 2])
            # grad_r *= 100000
            # # grad_ori = grad_ori.reshape(grad_ori.shape[0], -1)
            # # print(data_ori[0] == data[0])
            # # print(grad_ori[0] == grad[0])
            #
            #
            # # self.model.train()
            #
            # # input_gradient_loss = nn.MSELoss()(grad_r, grad)
            # mask_loss = nn.MSELoss()(mask_r, mask)
            # input_gradient_loss = nn.MSELoss()(grad_r, grad_r_r)
            # # loss = s_loss + us_loss * self.args.usvt_weight
            # # loss = us_loss * self.args.usvt_weight
            # # loss = s_loss
            #
            # # if self.cur_epoch != 0 and 20*input_gradient_loss.item() < s_loss.item():
            # #     loss = s_loss
            # # else:
            # #     loss = s_loss +  5*input_gradient_loss
            # #     # loss = s_loss
            # # if abs_grad.item()>0.4 and abs_grad.item() < 1.0:
            # #     loss = s_loss  +  5*input_gradient_loss
            # # else:
            # #     loss = s_loss
            loss = s_loss


            # loss = s_loss + 80 * input_gradient_loss
            loss.backward()
            self.optimizer.step()

            # record and print
            acc_s = torch.sum(c_l_pred == c_l).item() / data.shape[0]
            # acc_s = -1
            # acc_u = torch.sum(n_pred == n).item() / data.shape[0]
            acc_u = -1
            if i == 0:
                col_n = ceil(len(self.train_data_loader) / self.collect_per_batch)
                print(f'epoch:{self.cur_epoch}/{self.args.epochs};bs:{data.shape[0]};'
                      f'lr:{" ".join([str(lr) for lr in lrs])}; '
                      f'{len(self.train_data_loader)}/{self.collect_per_batch}={col_n}|', end='')
            if i % self.collect_per_batch == 0:
                print('#', end='')
                if self.args.wandb:
                    wandb.log({'acc/train/sv_task': acc_s,
                                'acc/train/usv_task': acc_u,
                                'loss/train/class': s_loss.item(),
                                # 'loss/train/usv_task': us_loss.item(),
                                'loss/train/sum': loss.item()})

            if i == len(self.train_data_loader) - 1:
                print()
                pp(f'@{acc_s}:train_acc:s;{acc_u}:u')
                pp(f'train_loss:s:{s_loss.item()}')
                # pp(f'mask_loss:{mask_loss.item()};input_gradient_loss:{input_gradient_loss.item()}')
                # pp(f'loss:{loss.item()},abs_grad:{abs_grad.item()},abs_grad_r:{abs_grad_r.item()}')

                # pp(f'train_loss:s:{s_loss.item()};u:{us_loss.item()}')
                # pp(f'@{acc_u}:train_acc:u')
                # pp(f'train_loss:u:{us_loss.item()}')

            # del loss, s_loss, us_loss, n_logit, c_l_logit
            torch.cuda.empty_cache()

        # eval
        self.model.eval()
        # with torch.no_grad():
        for phase, loader in self.test_s_loaders.items():
            s_acc, us_acc = Trainer.test(self.model, loader, device=self.device)
            pp(f'${s_acc}:{phase}_acc')
            if self.args.wandb:
                wandb.log({f'acc/{phase}': s_acc})
            self.results[phase][self.cur_epoch] = s_acc

            # for phase, loader in self.test_us_loaders.items():
            #     s_acc, us_acc = Trainer.test(self.model, loader, device=self.device)
            #     pp(f'${us_acc}:{phase}_acc')
            #     if self.args.wandb:
            #         wandb.log({f'acc/{phase}': us_acc})
            #     self.results[phase][self.cur_epoch] = us_acc



    @staticmethod
    def test(model, loader, device='cpu'):
        label_correct = 0
        n_correct = 0
        total = len(loader.dataset)
        for _, (data, n, c_l) in enumerate(loader):
            data, n, c_l = data.to(device), n.to(device), c_l.to(device)
            n_logit, c_l_logit = model(data)
            ## temporary used
            c_l_logit = n_logit
            _, c_l_pred = c_l_logit.max(dim=1)
            # _, n_pred = n_logit.max(dim=1)
            label_correct += torch.sum(c_l_pred == c_l).item()
            # n_correct += torch.sum(n_pred == n).item()
        return float(label_correct)/total, float(n_correct)/total


def iterate_args(args):
    args_list = list()
    for params in args.parameters:
        args.params = params
        args.usvt_weight = params[0]
        args.original_img_prob = params[1]

        s2ts = ms2st(args.domains, args.targets)
        for s2t in s2ts:
            args.source = s2t['s']
            args.target = s2t['t']
            for i in range(int(args.repeat_times)):
                args.nth_repeat = i
                args_list.append(copy.deepcopy(args))
    return args_list


def main():
    torch.autograd.set_detect_anomaly(True)
    # This flag allows you to enable the inbuilt cudnn auto-tuner to
    # find the best algorithm to use for your hardware.
    torch.backends.cudnn.benchmark = True

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    args = get_args()
    for args in iterate_args(args):
        output_dir = f'{args.output_dir}/{socket.gethostname()}/{args.experiment}/{args.network}/' + \
        '_'.join([str(_) for _ in args.params])+'/'
        writer = Writer(
            output_dir=output_dir,
            file=f'{args.source[0]}_{args.target}'
        )

        model = model_fns[args.network](
            num_usv_classes=args.num_usv_classes,
            num_classes=args.num_classes)
        if args.load_model != None:
            print(f"load model data/cache/222server/{args.load_model}")
            state_dict = torch.load(f'{os.path.dirname(__file__)}/'
                                    f'data/cache/222server/{args.load_model}.pkl')
            # del state_dict["classifier.fc6.weight"]
            # del state_dict["classifier.fc6.bias"]
            # del state_dict["classifier.fc7.weight"]
            # del state_dict["classifier.fc7.bias"]
            # del state_dict["usv_classifier.weight"]
            # del state_dict["usv_classifier.bias"]
            model.load_state_dict(state_dict, strict=False)

        # monitor_memory()

        if args.wandb:
            tags = [args.source[0] + '_' + args.target, "_".join([str(_) for _ in args.params])]
            wandb.init(project=f'{args.experiment}_{args.network}', tags=tags,
                       dir=dirname(__file__), config=args, reinit=True,
                       name=f'{"-".join([str(_) for _ in args.params])}-{args.source[0]}-{args.target}')
        if args.redirect_to_file and args.redirect_to_file != 'null':
            print('redirect to ', output_dir + args.redirect_to_file)
            sys.stdout = open(output_dir + args.redirect_to_file, 'a')
            # wandb.watch(model, log='all') # can lead to continuous increment of GPU memory usage

        data_loaders = get_DGSSR_data_loader(args.source, args.target, args.data_dir, args.val_size,
                                             args.original_img_prob, args.batch_size,
                                             args.max_num_s_img, args)
        optimizer = get_optimizer(model, lr=args.learning_rate, train_all=args.train_all_param)
        scheduler = optim.lr_scheduler.StepLR(optimizer, int(args.epochs * .8))
        Trainer(args, model, data_loaders, optimizer, scheduler, writer)

        save_model_dir = f'{args.data_dir}/cache/{socket.gethostname()}/{args.experiment}/{args.network}/' + \
                         '_'.join([str(_) for _ in args.params])
        if not os.path.exists(save_model_dir): os.makedirs(save_model_dir)

        torch.save(model.state_dict(), save_model_dir+f'/{args.source[0]}_{args.target}.pkl')

        # wandb.save(args.data_dir+'/cache/model.pkl')
        if args.wandb:
            wandb.join()
        #torch.save(model.state_dict(), output_dir + args.target)


if __name__ == "__main__":
    main()





