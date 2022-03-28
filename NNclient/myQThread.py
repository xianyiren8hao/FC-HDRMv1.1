from PyQt5.QtCore import QThread, pyqtSignal
from socket import AF_INET, SOCK_STREAM, socket
from Data_Pack import myPack, myRecv
from PIL import Image
from torch import save as T_save, load as T_load


class FC_HDRM_Work(QThread):
    # 连接信号
    sig_confail = pyqtSignal()
    sig_conok = pyqtSignal()
    sig_conoff = pyqtSignal()
    # 数据集信号
    sig_dsfail = pyqtSignal()
    sig_dsok = pyqtSignal()
    # 样本图信号
    sig_samfail = pyqtSignal()
    sig_samok = pyqtSignal()
    # 权重文件信号
    sig_wffail = pyqtSignal()
    sig_wfok = pyqtSignal()
    # 神经网络操作信号
    sig_nnofail = pyqtSignal()
    sig_nnook = pyqtSignal()
    # 神经网络中止信号
    sig_nnsfail = pyqtSignal()
    sig_nnsok = pyqtSignal()
    # 神经网络状态信号
    sig_nngsfail = pyqtSignal()
    sig_nngsok = pyqtSignal()
    # 神经网络数据信号
    sig_nngdfail = pyqtSignal()
    sig_nngdok = pyqtSignal()

    def __init__(self):
        super(FC_HDRM_Work, self).__init__()
        # 状态变量 - 线程控制
        self.thr_state = 0
        self.thr_e = None
        # 状态变量 - 网络连接
        self.tcp_on = False
        self.tcp_ip = None
        self.tcp_port = None
        self.tcp_text = None
        self.sc_c = socket(AF_INET, SOCK_STREAM)
        self.wsec_c = 15.0
        self.sc_c.settimeout(self.wsec_c)
        # 状态变量 - 数据读取
        self.dset_on = False
        self.dset_path = ""
        self.dset_strain = None
        self.dset_stest = None
        self.dset_ltrain = None
        self.dset_ltest = None
        # 状态变量 - 样本显示
        self.sam_index = None
        self.sam_count = None
        self.sam_fromS = False
        self.sam_img = None
        self.sam_str = ''
        # 状态变量 - 权重文件
        self.nn_wsave = False
        self.nn_wpath = ''
        self.nn_wfile = None
        # 状态变量 - 网络状态
        self.nn_workon = False
        self.nn_nowepoch = 0
        self.nn_nowbatch = 0
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
        self.nn_acur_ok = False
        self.nn_loss_ok = False
        self.sta_nums = None
        self.sta_cors = None
        self.sta_rates = None
        self.sta_losses = None
        self.sta_depoch = 1
        self.nn_onelabs = None
        self.nn_oneouts = None

    def run(self):
        # 0:初始化状态
        # 1:网络连接
        # 2:数据读取
        # 3:样本展示
        # 4:存取权重文件
        # 5:请求神经网络运作
        # 6:请求神经网络中止
        # 7:请求查看网络状态
        # 8:请求获取网络数据
        if self.thr_state == 1:
            self.Ser_con()
        elif self.thr_state == 2:
            self.Dset_init()
        elif self.thr_state == 3:
            self.Sam_get()
        elif self.thr_state == 4:
            self.Wfile_sl()
        elif self.thr_state == 5:
            self.NN_opa()
        elif self.thr_state == 6:
            self.NN_stop()
        elif self.thr_state == 7:
            self.NN_getstate()
        elif self.thr_state == 8:
            self.NN_getdata()

    def Ser_con(self):
        if self.tcp_on:
            self.tcp_text = {'cmd': 'close'}
            self.sc_c.sendall(myPack(self.tcp_text))
            self.sc_c.close()
            self.sc_c = socket(AF_INET, SOCK_STREAM)
            self.sc_c.settimeout(self.wsec_c)
            self.tcp_on = False
            self.sig_conoff.emit()
        else:
            try:
                self.sc_c.connect((self.tcp_ip, self.tcp_port))
            except Exception as e:
                self.thr_e = e
                self.tcp_on = False
                self.sig_confail.emit()
            else:
                self.tcp_on = True
                self.sig_conok.emit()

    def Dset_init(self):
        self.tcp_text = {'cmd': 'loaddset', 'droot': self.dset_path}
        self.sc_c.sendall(myPack(self.tcp_text))
        try:
            self.tcp_text = myRecv(self.sc_c)
            self.dset_on = self.tcp_text['dset_on']
        except Exception as e:
            self.thr_e = e
            self.dset_on = False
        else:
            self.thr_e = '服务端异常'
        if not self.dset_on:
            self.sig_dsfail.emit()
        else:
            self.dset_ltrain = self.tcp_text['ltrain']
            self.dset_ltest = self.tcp_text['ltest']
            self.sig_dsok.emit()

    def Sam_get(self):
        self.tcp_text = {'cmd': 'samget',
                        'index': self.sam_index,
                        'count': self.sam_count}
        self.sc_c.sendall(myPack(self.tcp_text))
        try:
            self.tcp_text = myRecv(self.sc_c)
            self.sam_img = self.tcp_text['img']
            self.sam_str = self.tcp_text['str']
        except Exception as e:
            self.thr_e = e
            self.sig_samfail.emit()
        else:
            self.sam_img = self.sam_img.resize((98, 98), Image.NEAREST)
            self.sam_img = self.sam_img.toqpixmap()
            self.sig_samok.emit()

    def Wfile_sl(self):
        if self.nn_wsave:
            self.tcp_text = {'cmd': 'swfile'}
            self.sc_c.sendall(myPack(self.tcp_text))
            try:
                self.tcp_text = myRecv(self.sc_c)
                self.nn_wfile = self.tcp_text['wfile']
                T_save(self.nn_wfile, self.nn_wpath)
            except Exception as e:
                self.thr_e = e
                self.sig_wffail.emit()
            else:
                self.sig_wfok.emit()
        else:
            try:
                self.nn_wfile = T_load(self.nn_wpath)
                self.tcp_text = {'cmd': 'lwfile', 'wfile': self.nn_wfile}
                self.sc_c.sendall(myPack(self.tcp_text))
                self.tcp_text = myRecv(self.sc_c)
                lw_ok = self.tcp_text['wl_ok']
            except Exception as e:
                self.thr_e = e
                lw_ok = False
            else:
                self.thr_e = '服务端异常'
            if not lw_ok:
                self.sig_wffail.emit()
            else:
                self.sig_wfok.emit()

    def NN_opa(self):
        self.tcp_text = {'cmd': 'nnopa',
                        'input': self.nn_input,
                        'needback': self.nn_needback,
                        'ifstatis': self.nn_ifstatis,
                        'epoch': self.nn_epoch,
                        'needloss': self.nn_needloss}
        self.sc_c.sendall(myPack(self.tcp_text))
        try:
            self.tcp_text = myRecv(self.sc_c)
            nno_ok = self.tcp_text['nno_ok']
            self.nn_workon = self.tcp_text['nn_workon']
        except Exception as e:
            self.thr_e = e
            nno_ok = False
        else:
            self.thr_e = '服务端已在工作中'
        if not nno_ok:
            self.sig_nnofail.emit()
        else:
            self.sig_nnook.emit()

    def NN_stop(self):
        self.tcp_text = {'cmd': 'nnstop'}
        self.sc_c.sendall(myPack(self.tcp_text))
        try:
            self.tcp_text = myRecv(self.sc_c)
            nns_ok = self.tcp_text['nns_ok']
            self.nn_workon = self.tcp_text['nn_workon']
        except Exception as e:
            self.thr_e = e
            nns_ok = False
        else:
            self.thr_e = '服务端未在工作中'
        if not nns_ok:
            self.sig_nnsfail.emit()
        else:
            self.sig_nnsok.emit()

    def NN_getstate(self):
        self.tcp_text = {'cmd': 'nngs'}
        self.sc_c.sendall(myPack(self.tcp_text))
        try:
            self.tcp_text = myRecv(self.sc_c)
            self.nn_workon = self.tcp_text['nn_workon']
            self.nn_nowepoch = self.tcp_text['nn_nowepoch']
            self.nn_nowbatch = self.tcp_text['nn_nowbatch']
        except Exception as e:
            self.thr_e = e
            self.sig_nngsfail.emit()
        else:
            self.sig_nngsok.emit()

    def NN_getdata(self):
        self.tcp_text = {'cmd': 'nngd'}
        self.sc_c.sendall(myPack(self.tcp_text))
        try:
            self.tcp_text = myRecv(self.sc_c)
            self.nn_acur_ok = self.tcp_text['nn_acur_ok']
            self.sta_nums = self.tcp_text['sta_nums']
            self.sta_cors = self.tcp_text['sta_cors']
            self.sta_rates = self.tcp_text['sta_rates']
            self.nn_loss_ok = self.tcp_text['nn_loss_ok']
            self.sta_losses = self.tcp_text['sta_losses']
            self.nn_one_ok = self.tcp_text['nn_one_ok']
            self.nn_onelabs = self.tcp_text['nn_onelabs']
            self.nn_oneouts = self.tcp_text['nn_oneouts']
            self.sta_depoch = self.tcp_text['sta_depoch']
        except Exception as e:
            self.thr_e = e
            self.sig_nngdfail.emit()
        else:
            self.sig_nngdok.emit()
