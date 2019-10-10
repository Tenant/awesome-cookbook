# QT Cheatsheet

## 1. File

加载配置文件

```c++
// Header
#include <QSettings>
#include <QFileInfo>
#include <QFileDialog>

// Cpp
QString ini_path = "config.ini";
while(!QFileInfo(ini_path).exists()) {
    ini_path = QFileDialog::getOpenFileName(nullptr, "Select config file", ".", "*.ini");
}
QSettings settings(ini_path,QSettings::IniFormat);
double xMin = settings.value("xMin",-6).toDouble();
double xMax = settings.value("xMax",6).toDouble();
double yMin = settings.value("yMin",-25).toDouble();
double yMax = settings.value("yMax",25).toDouble();
int segLength = settings.value("segLength",10).toInt();
double threshold = settings.value("threshold",-7).toDouble();
```
