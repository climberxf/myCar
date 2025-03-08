#include "qtudpyolo.h"
int showPictureCount = 0;
int picnum = 0;

QTUDPRecv1::QTUDPRecv1(QObject *parent) : QThread(parent)
{
    modelyolo = new ModelWrapper1("yolov7.param", "yolov7.bin");
    initUDP();
}
void QTUDPRecv1::run()
{
    showPicture();
}
void QTUDPRecv1::initUDP()
{
    ncnn::VulkanDevice vkdev(0);
    _qUdpSocket = new QUdpSocket(this);
    udpIP = "127.0.0.1";
    udpPort = 8888;
    qDebug() << "组播端口和IP为：" << udpPort << "  " << udpIP;
    _qUdpSocket->bind(QHostAddress::AnyIPv4, udpPort, QUdpSocket::ShareAddress);
    _qUdpSocket->setSocketOption(QAbstractSocket::ReceiveBufferSizeSocketOption, 1024 * 1024 * 8);
    connect(_qUdpSocket, SIGNAL(readyRead()), this, SLOT(recvDataSlot()));
}
cv::Mat QTUDPRecv1::QImage2Mat(QImage image)
{
    cv::Mat mat;
    switch (image.format())
    {
    case QImage::Format_RGB32:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
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

cv::Mat QTUDPRecv1::QImage2cvMat(const QImage &image)
{
    cv::Mat mat;
    switch (image.format())
    {
    case QImage::Format_Grayscale8:
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        break;
    case QImage::Format_RGBA64:
        mat = cv::Mat(image.height(), image.width(), CV_16UC4, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
        break;
    }
    return mat;
}

QTUDPRecv1::~QTUDPRecv1()
{
}

void QTUDPRecv1::recvDataSlot()
{
    std::chrono::steady_clock::time_point startTime;
    startTime = std::chrono::steady_clock::now();
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
            showPicture();
            auto endTime = std::chrono::steady_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            std::cout << "full time" << elapsedTime << "ms" << std::endl;
        }
        else if (datagram == "End!")
        {
//            qDebug() << "接收完成";
            break;
        }
        else
        {
            picBuffer.append(datagram);
        }
    }
}

//void QTUDPRecv1::openDirSlot()
//{
//    imagePath = "test001.jpg";
//    QFileInfo fileInfo(imagePath);
//    imageName = fileInfo.fileName();
//}

void QTUDPRecv1::showPicture()
{
    showPictureCount++;
    quint16 sendport = 8889;
    qDebug() << "showPicture() has been called " << showPictureCount << " times.";
    QBuffer buffer(&picBuffer);
    buffer.open(QIODevice::ReadOnly);
    QImageReader reader(&buffer);
    QImage img = reader.read();
//    picnum += 1;
//    QString picname = QString("image%1.jpg").arg(picnum);
    QString picname = QString("image001.jpg");
    img.save(picname);
    std::string imagename = picname.toStdString();
    cv::Mat mm = cv::imread(imagename);
    std::vector<Object1> objects;
    modelyolo->inference1(mm, objects);
    draw_objects(mm, objects);
    std::string objectsStr;
    for (const auto& obj : objects) {
        objectsStr += std::to_string(obj.label) + " " +
                      std::to_string(obj.prob) + " " +
                      std::to_string(obj.rect.x) + " " +
                      std::to_string(obj.rect.y) + " " +
                      std::to_string(obj.rect.width) + " " +
                      std::to_string(obj.rect.height) + " ";
    }
    QByteArray send_data = objectsStr.c_str();
    _qUdpSocket->writeDatagram(send_data, QHostAddress("127.0.0.1"), sendport);
//    qDebug() << "infor:" << objectsStr.c_str();

//    QByteArray send_endinfo = "end_yolo";
//    _qUdpSocket->writeDatagram(send_endinfo, QHostAddress("127.0.0.1"), sendport);
}


