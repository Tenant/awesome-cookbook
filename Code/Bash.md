**Recursively coping files with specified extension**

```bash
cp --parents -R folder_src/**/*.mp4 ./folder_dst
```



**Converting batches images into video**

```bash
ffmpeg -framerate 25 -i C%06d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
```



**Batch rename files**

```bash
ls *jpg | awk -F\. '{print "mv "$0" C"NR".jpg"}' | sh
```



**Empty Trash in Command Line**

```bash
rm -rf ~/.local/share/Trash/*
```

