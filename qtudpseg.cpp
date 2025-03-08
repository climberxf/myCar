#include "qtudpseg.h"
static int showPictureCount = 0;
QTUDPRecv::QTUDPRecv(QObject *parent) : QThread(parent)
{
    model = new ModelWrapper("yolopv2.param","yolopv2.bin");
    initUDP();
}

void QTUDPRecv::run()
{
    qDebug() <<"begin run";
    showPictureseg();
}

cv::Mat QTUDPRecv::QImage2Mat(QImage image)
{
    cv::Mat mat;
    switch (image.format())
    {
    case QImage::Format_RGB32:  //一般Qt读入彩色图后为此格式
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat,mat,cv::COLOR_BGRA2BGR);   //转3通道
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat,mat,cv::COLOR_RGB2BGR);
        break;
    case QImage::Format_Indexed8:
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_Invalid:
        break;
    default:
        break;
    }
    return mat;
}

cv::Mat QTUDPRecv::QImage2cvMat(const QImage &image)
{
    cv::Mat mat;
    switch(image.format())
    {
    case QImage::Format_Grayscale8: // 灰度图，每个像素点1个字节（8位）
        // Mat构造：行数，列数，存储结构，数据，step每行多少字节
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_ARGB32: // uint32存储0xAARRGGBB，pc一般小端存储低位在前，所以字节顺序就成了BGRA
    case QImage::Format_RGB32: // Alpha为FF
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888: // RR,GG,BB字节顺序存储
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        // opencv需要转为BGR的字节顺序
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        break;
    case QImage::Format_RGBA64: // uint64存储，顺序和Format_ARGB32相反，RGBA
        mat = cv::Mat(image.height(), image.width(), CV_16UC4, (void*)image.constBits(), image.bytesPerLine());
        // opencv需要转为BGRA的字节顺序
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
        break;
    }
    return mat;
}

QTUDPRecv::~QTUDPRecv()
{
    delete _qUdpSocket;
}

void QTUDPRecv::initUDP()
{
    //open gpu
    ncnn::VulkanDevice vkdev(0);
    //auto startTime = std::chrono::steady_clock::now();
    // 创建模型封装对象
//    model = new ModelWrapper("yolopv2.param","yolopv2.bin");
    // 创建QTUDPRecv对象并执行相应功能
    // 创建socket
    _qUdpSocket = new QUdpSocket(this);
    // 手动设置IP为固定的"127.0.0.1"
    udpIP = "127.0.0.1";
    // 获取Port（示例中不再获取UI元素的值，使用固定的端口12345）
    udpPort = 7777;
    qDebug() << "组播端口和IP为：" << udpPort << "  " << udpIP;
    // 绑定本地端口
    _qUdpSocket->bind(QHostAddress::AnyIPv4, udpPort, QUdpSocket::ShareAddress);
    // 设置缓冲区
    _qUdpSocket->setSocketOption(QAbstractSocket::ReceiveBufferSizeSocketOption, 1024 * 1024 * 8);
    // 当接收到数据，则进行处理
    connect(_qUdpSocket, SIGNAL(readyRead()), this, SLOT(recvDataSlot()));
}

void QTUDPRecv::recvDataSlot()
{
    while (_qUdpSocket->hasPendingDatagrams())
    {
        QByteArray datagram;
        datagram.resize(_qUdpSocket->pendingDatagramSize());
        QHostAddress sender;
        quint16 Pic_port;
        _qUdpSocket->readDatagram(datagram.data(), datagram.size(), &sender, &Pic_port);

        if (datagram == "Begin!")
        {
            picBuffer.clear();
        }
        else if (datagram == "continue!")
        {
            showPictureseg();
        }
        else if (datagram == "End!")
        {
//            qDebug()<<"接收完成";
            break;
        }
        else
        {
            picBuffer.append(datagram);
        }
    }
}

//void QTUDPRecv::openDirSlot()
//{
//    // 获取文件路径（示例中直接使用固定的文件路径）
//    imagePath = "test001.jpg";

//    // 获取文件名
//    QFileInfo fileInfo(imagePath);
//    imageName = fileInfo.fileName();
//}

void QTUDPRecv::showPictureseg()
{
    showPictureCount++;
    qDebug() << "showPicture(seg) has been called " << showPictureCount << " times.";
    QBuffer buffer(&picBuffer);
    buffer.open(QIODevice::ReadOnly);
    QImageReader reader(&buffer);
    QImage imgs = reader.read();
    QImage img = imgs.convertToFormat(QImage::Format_RGBA8888);
    qDebug()<<"img.size"<<img.format();
    cv::Mat mm(img.height(), img.width(), CV_8UC4, const_cast<uchar*>(img.constBits()), img.bytesPerLine());
    cv::cvtColor(mm, mm, cv::COLOR_BGR2RGB);
    ncnn::Mat da_seg_mask, ll_seg_mask;
    model->inference(mm,da_seg_mask,ll_seg_mask);
    cv::Mat mmm = draw_objects(mm, da_seg_mask,ll_seg_mask);
//    QImage image(mmm.data, mmm.cols, mmm.rows, static_cast<int>(mmm.step), QImage::Format_RGB888);


    QImage image(mmm.data, mmm.cols, mmm.rows, static_cast<int>(mmm.step), QImage::Format_RGB888);
    // 将图像从BGR颜色格式转换为RGB颜色格式
    image = image.rgbSwapped();


    QByteArray imageData;
    QBuffer buffer2(&imageData);
    buffer2.open(QIODevice::WriteOnly);
    image.save(&buffer2, "JPEG");//RGB_888
    buffer2.close();

    QHostAddress receiverAddress = QHostAddress("127.0.0.1");
    quint16 sendport = 7778;

    const int packetSize = 65000;

    int totalPackets = (imageData.size() + packetSize - 1) / packetSize;

    QByteArray begin = "Begin!";
    _qUdpSocket->writeDatagram(begin, QHostAddress("127.0.0.1"), sendport);
//    qDebug() << "发送：Begin!";

    for (int packetIndex = 0; packetIndex < totalPackets; ++packetIndex)
    {
        QByteArray packetData = imageData.mid(packetIndex * packetSize, packetSize);
        if (_qUdpSocket->writeDatagram(packetData, QHostAddress("127.0.0.1"), sendport) == -1)
        {
            qDebug() << "Failed to send packet " << packetIndex;
        }
//        QThread::msleep(5);
    }

    QByteArray end = "continue!";
    _qUdpSocket->writeDatagram(end, QHostAddress("127.0.0.1"), sendport);
//    qDebug() << "发送：continue!";
}
