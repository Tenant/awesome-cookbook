# Python

## 1. Pandas

## 2. Matplotlib



## 3. OS

`os.path.isfile('filename')`

```python
os.path.isfile('filename')
os.path.dirname(__file__)
```



## 4. Sys

```python
sys.path.append('path')
```

## 5. Pillow

**Draw Rectangle and Font on image**

```python
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

img = Image.open(file_name)

draw = ImageDraw.Draw(img)
draw.rectangle(((0, 0), (100, 100)), fill=None)
draw.text((20, 70), "DYP", font=ImageFont.truetype("font_path123"))

img.show()
img.save(out_file, "JPEG")
```
