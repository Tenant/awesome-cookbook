## Python

### 0. Basic

**Set default executable**

```python
#!/usr/bin/env python
```

**Function Definition**

```python
def function_name(val_, val_2):
    # do something
    return val_3
```

**Switch Structure**

```python
if expression_1 and expression_2:
    # do something
elif expression_3 or expression_4:
    # do something
else:
    # do something
```

**Compose and decompose list**

```python
similarities=[]
similarities.append(val_1)
similarities.append(val_2)

val_3, val_4=similarites
```

**Iterating all list**

```python
for i in range(val_1):
    # do something
```

**Reading and parsing files**

```python
fp=open(file_name)
lines=[line.rstrip('\n') for line in fp.readlines()]

for line in lines:
    line=line.replace('origin_string','target_string')
    print(line)
```

**Define the default entry function**

```python
def main(argv):

if __name__="__main__":
    main(sys.argv)
```



### 1. Pandas

### 2. Matplotlib



### 3. OS

`os.path.isfile('filename')`

```python
os.path.isfile('filename')
os.path.dirname(__file__)
```

**Check path existence**

```python
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
```



### 4. Sys

```python
sys.path.append('path')
```

### 5. Pillow

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

### 6. String

```python
list_of_lines=one_large_multiple_string.splitlines()
```

```python
one_large_string='\n'.join(list_of_lines)
```

**String to list**

```python
thelist = list(thestring)
```

```python
for c in thestring:
    do_something_with(c)
```

```python
results=[do_something_with(c) for c in thestring]
```

```python
results=map(do_something, the_string)
```

```python
import sets
magic_chars=sets.Set('abracadabra')
poppins_chars=set.Set('supercalifragilisticexpialidocious')
print(''.join(magic_chars & poppins_chars))
```

```python
print(ord('a'))
print(chr(97))
```



