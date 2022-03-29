from socket import socket, AF_INET, SO_REUSEADDR, SOCK_STREAM, SOL_SOCKET, timeout
from NNserver.myThread import FC_Para
from Data_Pack import myPack, myRecv


class myServer():
    def __init__(self, args):
        # 构造工作线程
        self.myPa = FC_Para()
        self.work_on = False
        # 构造连接套接字
        self.sc_s = socket(AF_INET, SOCK_STREAM)
        self.wsec_s = 120.0
        self.sc_c = None
        self.wsec_c = 60.0
        self.addr_c = None
        # 构造指令-函数映射关系
        self.cmd_dic = {
            'close': self.Work_close,
            'loaddset': self.Work_ldset,
            'samget': self.Work_samget,
            'swfile': self.Work_swfile,
            'lwfile': self.Work_lwfile,
            'nnopa': self.Work_nnopa,
            'nnstop': self.Work_nnstop,
            'nngs': self.Work_nngs,
            'nngd': self.Work_nngd
        }
        self.cmd_fun = None
        self.tcp_text = None
        # 绑定地址，连接客户端
        self.Con_wait(args)
    
    def Con_wait(self, args):
        self.sc_s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.sc_s.settimeout(self.wsec_s)
        try:
            self.sc_s.bind((args.host, args.port))
        except Exception as e:
            print(e)
            return
        self.sc_s.listen(1)
        print('--正在监听：{}'.format(self.sc_s.getsockname()))
        try:
            self.sc_c, self.addr_c = self.sc_s.accept()
        except Exception as e:
            print(e)
        else:
            print('收到连接：{}'.format(self.addr_c))
            self.sc_c.settimeout(self.wsec_c)
            self.work_on = True

    def Work(self):
        print('--正在等待客户端指令')
        try:
            self.tcp_text = myRecv(self.sc_c)
            self.cmd_fun = self.cmd_dic[self.tcp_text['cmd']]
        except timeout as e:
            print(e)
            return
        except Exception as e:
            print(type(e), e)
            self.sc_c.close()
            self.work_on = False
            return
        print('收到指令：{}'.format(self.tcp_text['cmd']))
        self.cmd_fun()

    def Work_close(self):
        self.sc_c.close()
        self.work_on = False

    def Work_ldset(self):
        self.myPa.dset_path = self.tcp_text['droot']
        self.myPa.Th_run(1)
        print('数据集加载中')
        while not self.myPa.th.thr_ok: pass
        if not self.myPa.dset_on:
            self.tcp_text = {'dset_on': False}
            print('数据集加载失败！')
        else:
            self.tcp_text = {'dset_on': True,
                            'strain': str(self.myPa.dset_train.data.size()),
                            'stest': str(self.myPa.dset_test.data.size()),
                            'ltrain': len(self.myPa.dset_train),
                            'ltest': len(self.myPa.dset_test)}
            print('数据集加载成功！')
        self.sc_c.sendall(myPack(self.tcp_text))

    def Work_samget(self):
        self.myPa.sam_index = self.tcp_text['index']
        self.myPa.sam_count = self.tcp_text['count']
        self.myPa.Th_run(2)
        print('正在获取样本图')
        while not self.myPa.th.thr_ok: pass
        self.tcp_text = {'img': self.myPa.sam_img, 'str': self.myPa.sam_str}
        print('样本图获取完成！')
        self.sc_c.sendall(myPack(self.tcp_text))

    def Work_swfile(self):
        self.myPa.nn_wsave = True
        self.myPa.Th_run(3)
        print('正在获取权重文件')
        while not self.myPa.th.thr_ok: pass
        self.tcp_text = {'wfile': self.myPa.nn_wfile}
        print('权重文件获取完成！')
        self.sc_c.sendall(myPack(self.tcp_text))

    def Work_lwfile(self):
        self.myPa.nn_wfile = self.tcp_text['wfile']
        self.myPa.nn_wsave = False
        self.myPa.Th_run(3)
        print('正在载入权重文件')
        while not self.myPa.th.thr_ok: pass
        self.tcp_text = {'wl_ok': self.myPa.nn_lw_ok}
        if self.myPa.nn_lw_ok:
            print('权重文件获取完成！')
        self.sc_c.sendall(myPack(self.tcp_text))

    def Work_nnopa(self):
        nno_ok = not self.myPa.nn_workon
        if nno_ok:
            print('神经网络开始运作')
            self.myPa.nn_input = self.tcp_text['input']
            self.myPa.nn_needback = self.tcp_text['needback']
            self.myPa.nn_ifstatis = self.tcp_text['ifstatis']
            self.myPa.nn_epoch = self.tcp_text['epoch']
            self.myPa.nn_needloss = self.tcp_text['needloss']
            self.myPa.Th_run(4)
        else:
            print('神经网络正在运作')
        self.tcp_text = {'nno_ok': nno_ok,
                        'nn_workon': self.myPa.nn_workon}
        self.sc_c.sendall(myPack(self.tcp_text))

    def Work_nnstop(self):
        nns_ok = self.myPa.nn_workon
        self.myPa.nn_stopnow = True
        if nns_ok:
            print('神经网络正在中止')
        else:
            print('神经网络未在运行')
        self.tcp_text = {'nns_ok': nns_ok,
                        'nn_workon': self.myPa.nn_workon}
        self.sc_c.sendall(myPack(self.tcp_text))

    def Work_nngs(self):
        self.tcp_text = {'nn_workon': self.myPa.nn_workon,
                        'nn_nowepoch': self.myPa.nn_nowepoch,
                        'nn_nowbatch': self.myPa.nn_nowbatch}
        self.sc_c.sendall(myPack(self.tcp_text))

    def Work_nngd(self):
        self.tcp_text = {'nn_acur_ok': self.myPa.nn_acur_ok,
                        'sta_nums': None,
                        'sta_cors': None,
                        'sta_rates': None,
                        'nn_loss_ok': self.myPa.nn_loss_ok,
                        'sta_losses': None,
                        'nn_one_ok': self.myPa.nn_one_ok,
                        'nn_onelabs': None,
                        'nn_oneouts': None,
                        'sta_depoch': self.myPa.nn_epoch}
        if self.myPa.nn_acur_ok:
            self.tcp_text['sta_nums'] = self.myPa.sta_nums
            self.tcp_text['sta_cors'] = self.myPa.sta_cors
            self.tcp_text['sta_rates'] = self.myPa.sta_rates
        if self.myPa.nn_loss_ok:
            self.tcp_text['sta_losses'] = self.myPa.sta_losses
        if self.myPa.nn_one_ok:
            self.tcp_text['nn_onelabs'] = self.myPa.nn_onelabs
            self.tcp_text['nn_oneouts'] = self.myPa.nn_oneouts
        self.sc_c.sendall(myPack(self.tcp_text))
