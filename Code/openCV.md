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

复制透明背景图片到另一个图片：
```c++
std::vector<cv::Mat> channels;
cv::split(src_img, channels);
mask = channels[3];
cv::Mat roi = dst_img(cv::Rect(center_x - src_img_width/2, center_y - src_img_height/2, src_img_width, sur_img_height));
src_img.copyTo(roi,mask);
```
**Create a Video**

```c++
cv::VideoWriter vid("outcpp2.avi",CV_FOURCC('M','J','P','G'),10, cv::Size(1200,1200));

cv::Mat img = imread("demo.png");
vid.write(img);

vid.release();
```

### 2. Python

**Install OpenCV-Python**

```bash
pip install opencv_python
```

