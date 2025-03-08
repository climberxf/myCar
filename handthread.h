#ifndef HANDTHREAD_H //#ifndef` 检查 `HANDTHREAD_H` 是否未定义，如果未定义，则包含 `#ifndef` 和 `#endif` 之间的代码。
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
    Q_OBJECT //宏是所有定义信号和槽并使用其他 Qt 特性的类所必需的。
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
