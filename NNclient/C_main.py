from PyQt5.QtWidgets import QMainWindow, QFileDialog
from NNclient.Ui_FCNNwin import Ui_FCNNwin
from NNclient.myQThread import FC_HDRM_Work
from Data_Pack import myPack


class MyMainWindow(QMainWindow, Ui_FCNNwin):
    def __init__(self, args, parent=None):
        # 窗口初始化
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        # 针对既定参数进行填入
        if len(args.host):
            self.LE_serverIP.setText(args.host)
        self.SB_serverPort.setValue(args.port)
        # 构建通信线程模型
        self.myWork = FC_HDRM_Work()
        # 连接线程信号
        self.myWork.sig_confail.connect(self.Con_failed)
        self.myWork.sig_conok.connect(self.Con_ok)
        self.myWork.sig_conoff.connect(self.Con_off)
        self.myWork.sig_dsfail.connect(self.Dset_failed)
        self.myWork.sig_dsok.connect(self.Dset_ok)
        self.myWork.sig_samfail.connect(self.Sam_failed)
        self.myWork.sig_samok.connect(self.Sam_ok)
        self.myWork.sig_wffail.connect(self.Wf_failed)
        self.myWork.sig_wfok.connect(self.Wf_ok)
        self.myWork.sig_nnofail.connect(self.NNopa_failed)
        self.myWork.sig_nnook.connect(self.NNopa_ok)
        self.myWork.sig_nnsfail.connect(self.NNstop_failed)
        self.myWork.sig_nnsok.connect(self.NNstop_ok)
        self.myWork.sig_nngsfail.connect(self.NNgs_failed)
        self.myWork.sig_nngsok.connect(self.NNgs_ok)
        self.myWork.sig_nngdfail.connect(self.NNgd_failed)
        self.myWork.sig_nngdok.connect(self.NNgd_ok)

    def closeEvent(self, event):
        if self.myWork.tcp_on:
            self.myWork.tcp_text = {'cmd': 'close'}
            self.myWork.sc_c.sendall(myPack(self.myWork.tcp_text))
            self.myWork.sc_c.close()
        event.accept()

    def Interact_ena(self, ena=True):
        # 线程工作时，禁止按键交互
        ena1 = ena and self.myWork.tcp_on
        ena2 = ena1 and self.myWork.dset_on
        ena3 = ena2 and (not self.myWork.nn_workon)
        ena4 = ena2 and (self.myWork.nn_acur_ok
                or self.myWork.nn_loss_ok or self.myWork.nn_one_ok)
        self.PB_connect.setEnabled(ena)
        self.PB_dload.setEnabled(ena1)
        self.CB_setchoice.setEnabled(ena3)
        self.SB_imagecount.setEnabled(ena3)
        self.PB_swfile.setEnabled(ena3)
        self.PB_lwfile.setEnabled(ena3)
        # 网络相关交互
        self.PB_nnopa.setEnabled(ena2)
        self.PB_nnstop.setEnabled(ena2)
        self.PB_getstate.setEnabled(ena2)
        self.PB_getdata.setEnabled(ena2)
        self.SB_depoch.setEnabled(ena4)

    def Con_clicked(self):
        self.Interact_ena(False)
        if self.myWork.tcp_on:
            self.LE_state.setText('正在断开……')
        else:
            self.LE_state.setText('正在连接……')
            self.LE_serverIP.setEnabled = False
            self.SB_serverPort.setEnabled = False
            self.myWork.tcp_ip = self.LE_serverIP.text()
            self.myWork.tcp_port = self.SB_serverPort.value()
        self.myWork.thr_state = 1
        self.myWork.start()

    def Con_failed(self):
        self.LE_serverIP.setEnabled = True
        self.SB_serverPort.setEnabled = True
        self.LE_state.setText(str(self.myWork.thr_e))
        self.LE_state.home(False)
        self.Interact_ena(True)

    def Con_ok(self):
        self.PB_connect.setText('断开')
        self.LE_state.setText('与服务端已建立连接')
        self.Interact_ena(True)

    def Con_off(self):
        self.PB_connect.setText('连接')
        self.LE_serverIP.setEnabled = True
        self.SB_serverPort.setEnabled = True
        self.LE_state.setText('已断开与服务端的连接')
        self.Interact_ena(True)

    def Dload_clicked(self):
        self.Interact_ena(False)
        self.LE_state.setText('正在发送载入指令……')
        self.myWork.dset_path = self.LE_droot.text()
        self.myWork.thr_state = 2
        self.myWork.start()

    def Dset_failed(self):
        self.LE_state.setText(str(self.myWork.thr_e))
        self.LE_state.home(False)
        self.Interact_ena(True)

    def Dset_ok(self):
        self.LE_dtrain.setText(self.myWork.tcp_text['strain'])
        self.LE_dtest.setText(self.myWork.tcp_text['stest'])
        self.LE_state.setText('数据集载入成功！')
        self.Schoice_activated(self.CB_setchoice.currentIndex())

    def Schoice_activated(self, dsindex):
        self.Interact_ena(False)
        if dsindex == 0:
            self.SB_imagecount.setMaximum(self.myWork.dset_ltrain - 1)
        elif dsindex == 1:
            self.SB_imagecount.setMaximum(self.myWork.dset_ltest - 1)
        self.myWork.sam_index = dsindex
        self.myWork.sam_count = self.SB_imagecount.value()
        self.myWork.thr_state = 3
        self.myWork.start()

    def Icount_vchanged(self, dscount):
        self.Interact_ena(False)
        self.myWork.sam_index = self.CB_setchoice.currentIndex()
        self.myWork.sam_count = dscount
        self.myWork.sam_fromS = True
        self.myWork.thr_state = 3
        self.myWork.start()

    def Sam_failed(self):
        self.LE_state.setText(str(self.myWork.thr_e))
        self.LE_state.home(False)
        self.Interact_ena(True)
        if self.myWork.sam_fromS:
            self.myWork.sam_fromS = False
            self.SB_imagecount.setFocus()

    def Sam_ok(self):
        self.L_image.setPixmap(self.myWork.sam_img)
        self.LE_sampletar.setText(self.myWork.sam_str)
        self.Interact_ena(True)
        if self.myWork.sam_fromS:
            self.myWork.sam_fromS = False
            self.SB_imagecount.setFocus()

    def Sdir_clicked(self):
        wfile, _ = QFileDialog.getSaveFileName(
            self, '选择保存位置', './', '所有文件(*.*)')
        self.LE_swfile.setText(wfile)

    def Swfile_clicked(self):
        self.Interact_ena(False)
        self.LE_state.setText('正在请求权重文件……')
        self.myWork.nn_wsave = True
        self.myWork.nn_wpath = self.LE_swfile.text()
        self.myWork.thr_state = 4
        self.myWork.start()

    def Ldir_clicked(self):
        wfile, _ = QFileDialog.getOpenFileName(
            self, '选择读取位置', './', '所有文件(*.*)')
        self.LE_lwfile.setText(wfile)

    def Lwfile_clicked(self):
        self.Interact_ena(False)
        self.LE_state.setText('正在读取权重文件……')
        self.myWork.nn_wsave = False
        self.myWork.nn_wpath = self.LE_lwfile.text()
        self.myWork.thr_state = 4
        self.myWork.start()

    def Wf_failed(self):
        self.LE_state.setText(str(self.myWork.thr_e))
        self.LE_state.home(False)
        self.Interact_ena(True)

    def Wf_ok(self):
        if self.myWork.nn_wsave:
            self.LE_state.setText('权重文件保存成功！')
        else:
            self.LE_state.setText('权重文件读取成功！')
        self.Interact_ena(True)

    def NNopa_clicked(self):
        self.Interact_ena(False)
        self.LE_nnstate.setText('正在请求神经网络运作……')
        self.myWork.nn_input = self.CB_nninput.currentIndex()
        self.myWork.nn_needback = self.CHB_nnbp.isChecked()
        self.myWork.nn_ifstatis = self.CHB_nnsta.isChecked()
        self.myWork.nn_epoch = self.SB_nnepoch.value()
        self.myWork.nn_needloss = self.CHB_nnloss.isChecked()
        self.myWork.thr_state = 5
        self.myWork.start()

    def NNopa_failed(self):
        self.LE_nnstate.setText(str(self.myWork.thr_e))
        self.LE_nnstate.home(False)
        self.Interact_ena(True)
    
    def NNopa_ok(self):
        self.LE_nnstate.setText('神经网络已开始运作！')
        self.Interact_ena(True)

    def NNstop_clicked(self):
        self.Interact_ena(False)
        self.LE_nnstate.setText('正在请求神经网络中止……')
        self.myWork.thr_state = 6
        self.myWork.start()

    def NNstop_failed(self):
        self.LE_nnstate.setText(str(self.myWork.thr_e))
        self.LE_nnstate.home(False)
        self.Interact_ena(True)

    def NNstop_ok(self):
        self.LE_nnstate.setText('已下达中止命令！')
        self.Interact_ena(True)

    def NNgs_clicked(self):
        self.Interact_ena(False)
        self.LE_nnstate.setText('正在获取网络状态……')
        self.myWork.thr_state = 7
        self.myWork.start()

    def NNgs_failed(self):
        self.LE_nnstate.setText(str(self.myWork.thr_e))
        self.LE_nnstate.home(False)
        self.Interact_ena(True)

    def NNgs_ok(self):
        self.LE_nnstate.setText('工作中: {}, epoch: {:0>4d}, batch: {:0>4d}'.format(
            self.myWork.nn_workon, self.myWork.nn_nowepoch + 1, self.myWork.nn_nowbatch + 1))
        self.Interact_ena(True)

    def NNgd_clicked(self):
        self.Interact_ena(False)
        self.LE_nnstate.setText('正在获取网络数据……')
        self.myWork.thr_state = 8
        self.myWork.start()

    def NNgd_failed(self):
        self.LE_nnstate.setText(str(self.myWork.thr_e))
        self.LE_nnstate.home(False)
        self.Interact_ena(True)

    def NNgd_ok(self):
        self.SB_depoch.setMaximum(self.myWork.sta_depoch)
        if self.myWork.nn_one_ok:
            self.TWonce_show()
        if self.myWork.nn_acur_ok or self.myWork.nn_loss_ok:
            self.Depoch_vchanged(self.SB_depoch.value())
        self.LE_nnstate.setText('Accuracy数据: {}, Loss数据: {}'.format(
            self.myWork.nn_acur_ok, self.myWork.nn_loss_ok))
        self.Interact_ena(True)

    def TWonce_show(self):
        for i in range(10):
            self.TW_nnonce.item(0, i).setText('{:.4f}'.format(self.myWork.nn_onelabs[i]))
            self.TW_nnonce.item(1, i).setText('{:.4f}'.format(self.myWork.nn_oneouts[i]))

    def Depoch_vchanged(self, decount):
        if self.myWork.nn_acur_ok:
            for i in range(11):
                self.TW_nnacr.item(0, i).setText('{:d}'.format(self.myWork.sta_nums[decount - 1][i]))
                self.TW_nnacr.item(1, i).setText('{:d}'.format(self.myWork.sta_cors[decount - 1][i]))
                self.TW_nnacr.item(2, i).setText('{:.2f}%'.format(self.myWork.sta_rates[decount - 1][i] * 100.00))
        if self.myWork.nn_loss_ok:
            self.LE_nnloss.setText('{:.4f}'.format(self.myWork.sta_losses[decount - 1]))
