from argparse import ArgumentParser


def NN_server(args):
    from NNserver.S_main import myServer
    mySer = myServer(args)
    while mySer.work_on:
        mySer.Work()
    print('--服务端已关闭')
    mySer.sc_s.close()


def QT_client(args):
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QCoreApplication, Qt
    from NNclient.C_main import MyMainWindow
    import sys
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    myWin = MyMainWindow(args)
    myWin.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    NN_roles = {'client': QT_client, 'server': NN_server}
    parser = ArgumentParser(description='基于MNIST数据集的FCNN手写识别方案')
    parser.add_argument('role', metavar='role', choices=NN_roles, default='client',
                        help='选择启动类型：client客户端；server服务端')
    parser.add_argument('--host', metavar='IP', default='',
                        help='服务端监听地址；客户端连接地址')
    parser.add_argument('--port', type=int, default=51015,
                        help='服务端监听端口；客户端连接端口')
    args = parser.parse_args()
    F_main = NN_roles[args.role]
    F_main(args)
