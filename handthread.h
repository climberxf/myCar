#ifndef HANDTHREAD_H //#ifndef` ��� `HANDTHREAD_H` �Ƿ�δ���壬���δ���壬����� `#ifndef` �� `#endif` ֮��Ĵ��롣
#define HANDTHREAD_H
#include <QtNetwork>//
#include <QObject>
#include <QtCore/QCoreApplication>
#include <QtNetwork/QUdpSocket>
#include <QDebug>
#include <QFile>
#include <QBuffer>
#include <QImageReader>
#include <QPixmap>
#include <QFileInfo>
#include <opencv2/imgproc/types_c.h>
#include <QImage>
#include <ncnn_seg.h>
#include <ncnn_yolo.h>
#include <qtudpseg.h>
#include <qtudpyolo.h>

class handthread : public QObject
{
    Q_OBJECT //�������ж����źźͲ۲�ʹ������ Qt ���Ե���������ġ�
public:
    handthread(QObject *parent = nullptr);
    void receivepic();
    void initUDP();
    void showPicture();
private:
    QUdpSocket * udpsocket;
    QString udpIP;
    quint16 udpport;
    QByteArray picbuffer;
};
#endif // HANDTHREAD_H
