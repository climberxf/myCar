#include "qtudpyolo.h"
#include <QtWidgets/QApplication>
#include <ncnn_yolo.h>
#include <ncnn_seg.h>
#include <qtudpseg.h>
#include <handthread.h>
#include <QThread>
#include <handthread.h>
#pragma execution_character_set("utf-8")
int flag_yolo = 0;
int flag_seg = 0;
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

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

    return a.exec();
}
