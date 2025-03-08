#ifndef QTUDPSEG_H
#define QTUDPSEG_H

#include <QObject>
#include <QtCore/QCoreApplication>
#include <QtNetwork/QUdpSocket>
#include <QDebug>
#include <QFile>
#include <QBuffer>
#include <QImageReader>
#include <QPixmap>
#include <QFileInfo>
#include <ncnn_seg.h>
#include <opencv2/imgproc/types_c.h>
#include <QImage>
#include <QThread>
#include <QImageReader>
class ModelWrapper;

class QTUDPRecv:public QThread
{
    Q_OBJECT
public:
    QTUDPRecv(QObject *parent = nullptr);
    void run() override;
    cv::Mat QImage2Mat(QImage image);
    cv::Mat QImage2cvMat(const QImage &image);

    ~QTUDPRecv();

public slots:
    void initUDP();
    void recvDataSlot();
    void showPictureseg();

private:
    QUdpSocket *_qUdpSocket;
    QString udpIP;
    quint16 udpPort;
    QString imagePath;
    QString imageName;
    QByteArray picBuffer;
    ModelWrapper * model;
};

#endif // QTUDPSEG_H
