#include "handthread.h"

handthread::handthread(QObject *parent) : QObject(parent)
{
    initUDP();
}
void handthread::initUDP()
{
    //将qtudpyolo的类移动到检测线程
    QThread model;
    QTUDPRecv modelyolo;
    modelyolo.moveToThread(&model);
    model.start();

    //将qtudpseg的类移动到分割线程
    QThread model1;
    QTUDPRecv1 modelseg;
    modelseg.moveToThread(&model1);
    model1.start();
}

//void handthread::initUDP()
//{
//    // 创建socket
//    udpsocket = new QUdpSocket();
//    //open gpu
//    ncnn::VulkanDevice vkdev(0);
//    // 手动设置IP为固定的"127.0.0.1"
//    udpIP = "127.0.0.1";
//    // 获取Port（示例中不再获取UI元素的值，使用固定的端口12345）
//    udpport = 8888;
//    qDebug() << "组播端口和IP为：" << udpport << "  " << udpIP;
//    // 绑定本地端口
//    udpsocket->bind(QHostAddress::AnyIPv4, udpport, QUdpSocket::ShareAddress);
//    qDebug() << "设置缓冲区";
//    // 设置缓冲区
//    udpsocket->setSocketOption(QAbstractSocket::ReceiveBufferSizeSocketOption, 1024 * 1024 * 8);
//    // 当接收到数据，则进行处理
//    connect(udpsocket, &QUdpSocket::readyRead, this, &handthread::receivepic);
//}
//void handthread::receivepic()
//{
//    while (udpsocket->hasPendingDatagrams())
//    {
//        QByteArray datagram;
//        datagram.resize(udpsocket->pendingDatagramSize());
//        QHostAddress sender;
//        quint16 Pic_port;
//        udpsocket->readDatagram(datagram.data(), datagram.size(), &sender, &Pic_port);

//        if (datagram == "Begin!")
//        {
//            picbuffer.clear();
//        }
//        else if (datagram == "continue!")
//        {
//            showPicture();
//        }
//        else if (datagram == "End!")
//        {
//            qDebug()<<"接收完成";
//            break;
//        }
//        else
//        {
//            picbuffer.append(datagram);
//        }
//    }
//}

//void handthread::showPicture()
//{
//    QBuffer buffer(&picbuffer);
//    buffer.open(QIODevice::ReadOnly);
//    QImageReader reader(&buffer);
//    QImage imgs = reader.read();
//    QImage img = imgs.convertToFormat(QImage::Format_RGBA8888);
//    img.save("image001.jpg");

//    //将qtudpseg的类移动到线程中
//    QThread model;
//    QTUDPRecv threadseg;
//    threadseg.moveToThread(&model);
//    model.start();

//    //将qtudpyolo的类移动到分割线程
//    QThread model1;
//    QTUDPRecv1 threadyolo;
//    threadyolo.moveToThread(&model1);
//    model1.start();
//}




