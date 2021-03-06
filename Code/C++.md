# Cpp Cheatsheet

## 1. Intro

[Cplusplus Offical Tutorial](http://www.cplusplus.com/doc/tutorial/)



## 2. Case Study

### 友元

[C++ friend 用法总结](https://blog.csdn.net/ddupd/article/details/38053159)

友元的作用在于可以将类的属性开放给友元的同时禁止其它类的访问。



### 智能指针

[shared_ptr(共享指针)使用总结](https://blog.csdn.net/wdxin1322/article/details/23738593)



### 多线程

[C++ 多线程](http://www.runoob.com/cplusplus/cpp-multithreading.html)



### STL

[C++ STL 教程](http://www.runoob.com/cplusplus/cpp-stl-tutorial.html)

[STL中set使用方法详细](https://blog.csdn.net/changjiale110/article/details/79108447)



### File

[fscanf格式详解](https://blog.csdn.net/q_l_s/article/details/22572777)

读取已知格式的文本文件：

```c++
FILE *fp;

while(true){
    int cnt = fscanf(fp, "%*d %*f %*f %lf %lf %lf %*[^\n]",yaw,gx,gy);
    if(cnt < 3){
        break;
    }
}
fclose(fp);
```

按照定义格式写入文本文件，每次写入前打开，写入后关闭：

```c++
FILE *fout = fopen("report.csv","a");
fprintf(fout,"%lf,%lf,%lf\n",std::get<1>(oneEnvCar.Traj[0]),std::get<2>(oneEnvCar.Traj[0]),n_path_deviation);
fclose(fout);
```

基于Qt创建目录

```c++
#include <QDir>

QDir *qDir = new QDir;
if (!qDir->exists("dataset")) {
    qDir->mkdir("dataset");
}

int frameCounter = 10;
QString path = QString("dataset/") + QString::number(frameCounter,10);
std::printf(path.toStdString().c_str());
if (!qDir->exists(path)) {
    qDir->mkdir(path);
}
```

### Timing

```c++
double time=getTickCount();

double seconds = ( (double)getTickCount() - time)/ getTickFrequency();
int days = int(seconds) / 60 / 60 / 24;
int hours = (int(seconds) / 60 / 60) % 24;
int minutes = (int(seconds) / 60) % 60;
int seconds_left = int(seconds) % 60;
cout << "Training until now has taken " << days << " days " << hours << " hours " << minutes << " minutes " << seconds_left <<" seconds." << endl;
```
