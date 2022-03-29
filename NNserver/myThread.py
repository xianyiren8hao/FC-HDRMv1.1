from threading import Thread
from torchvision.transforms import Compose, ToTensor, ToPILImage
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from NNserver.myFCNN import FC_HDRM_Net
from torch.nn.functional import softmax


DL_BSIZE = 1000


class FC_Para():
    def __init__(self):
        # 状态变量 - 数据读取
        self.dset_on = False
        self.dset_path = ""
        self.dset_trans = Compose([ToTensor()])
        self.dset_train = None
        self.dset_test = None
        self.dload_train = None
        self.dload_test = None
        # 状态变量 - 样本显示
        self.sam_index = None
        self.sam_count = None
        self.sam_trans = Compose([ToPILImage()])
        self.sam_tsr = None
        self.sam_img = None
        self.sam_tar = None
        self.sam_str = ''
        # 状态变量 - 权重文件
        self.nn_wsave = False
        self.nn_wfile = None
        self.nn_lw_ok = False
        # 状态变量 - 网络状态
        self.nn_workon = False
        self.nn_nowepoch = 0
        self.nn_nowbatch = 0
        self.nn_stopnow = False
        self.nn_acur_ok = False
        self.nn_loss_ok = False
        self.nn_one_ok = False
        # 状态变量 - 工作参数
        self.nn_input = 0
        self.nn_needback = False
        self.nn_ifstatis = False
        self.nn_epoch = 1
        self.nn_needloss = False
        # 状态变量 - 网络数据
        self.nn_net = FC_HDRM_Net().cuda()
        self.nn_lr = 0.0005
        self.nn_opt = torch.optim.SGD(self.nn_net.parameters(), self.nn_lr * DL_BSIZE)
        self.nn_inputs = None
        self.nn_inlabs = None
        self.nn_outputs = None
        self.nn_onelabs = None
        self.nn_oneouts = None
        # 状态变量 - 网络数据统计
        self.sta_predict = None
        self.sta_correct = None
        self.sta_epo_num = None
        self.sta_epo_cor = None
        self.sta_epo_rate = None
        self.sta_nums = None
        self.sta_cors = None
        self.sta_rates = None
        self.sta_bth_loss = None
        self.sta_epo_loss = None
        self.sta_losses = None
        # 线程调度
        self.th = None

    def Th_run(self, ths):
        self.th = FC_Thread(self, ths)
        self.th.start()


class FC_Thread(Thread):
    def __init__(self, obj, ths):
        super(FC_Thread, self).__init__()
        # 线程控制
        self.p = obj
        self.thr_state = ths
        self.thr_ok = False

    def run(self):
        # 0:初始化状态
        # 1:数据读取
        # 2:样本展示
        # 3:存取权重文件
        # 4:神经网络运行
        if self.thr_state == 1:
            self.Dset_init()
        elif self.thr_state == 2:
            self.Sam_get()
        elif self.thr_state == 3:
            self.Wfile_sl()
        elif self.thr_state == 4:
            self.NN_opa()
        self.thr_ok = True

    def Dset_init(self):
        try:
            self.p.dset_train = MNIST(root=self.p.dset_path, train=True,
                                    transform=self.p.dset_trans, download=True)
            self.p.dset_test = MNIST(root=self.p.dset_path, train=False,
                                    transform=self.p.dset_trans, download=True)
        except Exception as e:
            print(e)
            self.p.dset_on = False
            return
        self.p.dload_train = DataLoader(
            self.p.dset_train, batch_size=DL_BSIZE, shuffle=True, drop_last=False)
        self.p.dload_test = DataLoader(
            self.p.dset_test, batch_size=DL_BSIZE, shuffle=False, drop_last=False)
        self.p.dset_on = True

    def Sam_get(self):
        if self.p.sam_index == 0:
            self.p.sam_tsr, self.p.sam_tar = self.p.dset_train[self.p.sam_count]
        elif self.p.sam_index == 1:
            self.p.sam_tsr, self.p.sam_tar = self.p.dset_test[self.p.sam_count]
        self.p.sam_img = self.p.sam_trans(self.p.sam_tsr)
        self.p.sam_tsr = self.p.sam_tsr.cuda()
        self.p.sam_tar = torch.tensor(self.p.sam_tar).cuda()
        self.p.sam_str = self.p.dset_train.classes[self.p.sam_tar]

    def Wfile_sl(self):
        if self.p.nn_wsave:
            self.p.nn_wfile = self.p.nn_net.state_dict()
        else:
            try:
                self.p.nn_net.load_state_dict(self.p.nn_wfile)
            except Exception as e:
                print(e)
                self.p.nn_lw_ok = False
            else:
                self.p.nn_lw_ok = True

    def NN_opa(self):
        self.p.nn_workon = True
        self.p.nn_stopnow = False
        self.p.nn_acur_ok = False
        self.p.nn_loss_ok = False
        self.p.nn_one_ok = False
        self.p.sta_nums = []
        self.p.sta_cors = []
        self.p.sta_rates = []
        self.p.sta_losses = []
        for self.p.nn_nowepoch in range(self.p.nn_epoch):
            # 确定迭代器
            if self.p.nn_input == 0:
                loader_iter = enumerate([0], 0)
            elif self.p.nn_input == 1:
                loader_iter = enumerate(self.p.dload_train, 0)
            elif self.p.nn_input == 2:
                loader_iter = enumerate(self.p.dload_test, 0)
            # 准备数据统计
            if self.p.nn_ifstatis:
                self.p.sta_epo_num = torch.zeros(11, dtype=torch.int).cuda()
                self.p.sta_epo_cor = torch.zeros(11, dtype=torch.int).cuda()
            if self.p.nn_needloss:
                self.p.sta_epo_loss = torch.zeros(1).cuda()
            for self.p.nn_nowbatch, datas in loader_iter:
                # 消息打印
                # print('epoch:{:0>4d},batch:{:0>4d}'.format(self.p.nn_nowepoch, self.p.nn_nowbatch))
                # 载入初始数据
                if self.p.nn_input == 0:
                    self.p.nn_inputs = Variable(self.p.sam_tsr.unsqueeze(0))
                    self.p.nn_inlabs = Variable(self.p.sam_tar.unsqueeze(0))
                else:
                    self.p.nn_inputs, self.p.nn_inlabs = datas
                    self.p.nn_inputs = Variable(self.p.nn_inputs.cuda())
                    self.p.nn_inlabs = Variable(self.p.nn_inlabs.cuda())
                # 前向与后向
                if self.p.nn_needback:
                    self.p.nn_opt.param_groups[0]['lr'] = self.p.nn_lr * self.p.nn_inputs.size(dim=0)
                    self.p.nn_opt.zero_grad()
                self.p.nn_outputs = self.p.nn_net(self.p.nn_inputs)
                if self.p.nn_needback:
                    self.p.sta_bth_loss = self.p.nn_net.loss(self.p.nn_outputs, self.p.nn_inlabs)
                    self.p.sta_bth_loss.backward()
                    self.p.nn_opt.step()
                # batch数据统计
                if self.p.nn_ifstatis:
                    self.p.sta_predict = torch.max(self.p.nn_outputs, 1).indices
                    self.p.sta_correct = (self.p.sta_predict == self.p.nn_inlabs)
                    for i in range(10):
                        self.p.sta_epo_num[i] += (self.p.nn_inlabs == i).sum()
                        self.p.sta_epo_cor[i] += (self.p.sta_correct * (self.p.nn_inlabs == i)).sum()
                if self.p.nn_needloss:
                    self.p.sta_epo_loss += self.p.sta_bth_loss
                # 检查中止信号
                if self.p.nn_stopnow:
                    self.p.nn_stopnow = False
                    self.p.nn_workon = False
                    return
            # epoch结尾
            if self.p.nn_input == 0:
                self.p.nn_onelabs = torch.zeros(10)
                self.p.nn_onelabs[self.p.sam_tar] = 1.0
                self.p.nn_onelabs = self.p.nn_onelabs.numpy().tolist()
                self.p.nn_oneouts = self.p.nn_outputs.cpu().detach().squeeze(0)
                self.p.nn_oneouts = softmax(self.p.nn_oneouts, dim=0)
                self.p.nn_oneouts = self.p.nn_oneouts.numpy().tolist()
            if self.p.nn_ifstatis:
                self.p.sta_epo_num[10] = self.p.sta_epo_num.sum()
                self.p.sta_epo_cor[10] = self.p.sta_epo_cor.sum()
                self.p.sta_epo_rate = self.p.sta_epo_cor / self.p.sta_epo_num
                self.p.sta_epo_num = self.p.sta_epo_num.cpu().numpy().tolist()
                self.p.sta_epo_cor = self.p.sta_epo_cor.cpu().numpy().tolist()
                self.p.sta_epo_rate = self.p.sta_epo_rate.cpu().numpy().tolist()
                self.p.sta_nums.append(self.p.sta_epo_num)
                self.p.sta_cors.append(self.p.sta_epo_cor)
                self.p.sta_rates.append(self.p.sta_epo_rate)
            if self.p.nn_needloss:
                self.p.sta_epo_loss /= (self.p.nn_nowbatch + 1)
                self.p.sta_epo_loss = self.p.sta_epo_loss.cpu().detach().item()
                self.p.sta_losses.append(self.p.sta_epo_loss)
        # 完成工作
        if self.p.nn_ifstatis:
            self.p.nn_acur_ok = True
        if self.p.nn_needloss:
            self.p.nn_loss_ok = True
        if self.p.nn_input == 0:
            self.p.nn_one_ok = True
        self.p.nn_workon = False
