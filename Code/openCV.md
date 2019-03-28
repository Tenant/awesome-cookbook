### 1. Cpp

**Initialize an `cv::Mat`**

```c++
cv::Mat img;
img = cv::Mat::zeros(cv::Size(width, height),CV_8UC3);
```



**Copy an `cv::Mat` inside an ROI of another one**

```c++
src.copyTo(dst(Rect(left, top, src.cols, src.rows)));
```



```c++
Mat dst_roi = dst(Rect(left, top, src.cols, src.rows));
src.copyTo(dst_roi);
```



### 2. Python

**Install OpenCV-Python**

```bash
pip install opencv_python
```

