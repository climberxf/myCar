#ifndef QTUDPYOLO_H
#define QTUDPYOLO_H

#include <QObject>
#include <QtCore/QCoreApplication>
#include <QtNetwork/QUdpSocket>
#include <QDebug>
#include <QFile>
#include <QBuffer>
#include <QImageReader>
#include <QPixmap>
#include <QFileInfo>
#include <ncnn_yolo.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>
#include <QThread>

class ModelWrapper1;

class QTUDPRecv1 : public QThread
{
    Q_OBJECT
public:
    QTUDPRecv1(QObject *parent = nullptr);
    void run() override;
    cv::Mat QImage2Mat(QImage image);
    cv::Mat QImage2cvMat(const QImage &image);

    ~QTUDPRecv1();

public slots:
    void initUDP();
    void recvDataSlot();
    void showPicture();

private:
    QUdpSocket *_qUdpSocket;
    QString udpIP;
    quint16 udpPort;
    QString imagePath;
    QString imageName;
    QByteArray picBuffer;
    ModelWrapper1 * modelyolo;
};

#endif // QTUDPYOLO_H
