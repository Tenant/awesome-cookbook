### 1. Intro

[Cplusplus Offical Tutorial](http://www.cplusplus.com/doc/tutorial/)



### 2. Case Study

**友元**

[C++ friend 用法总结](https://blog.csdn.net/ddupd/article/details/38053159)

友元的作用在于可以将类的属性开放给友元的同时禁止其它类的访问。



**智能指针**

[shared_ptr(共享指针)使用总结](https://blog.csdn.net/wdxin1322/article/details/23738593)



**多线程**

[C++ 多线程](http://www.runoob.com/cplusplus/cpp-multithreading.html)



**STL**

[C++ STL 教程](http://www.runoob.com/cplusplus/cpp-stl-tutorial.html)



**File**

[fscanf格式详解](https://blog.csdn.net/q_l_s/article/details/22572777)

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

