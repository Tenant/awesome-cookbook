# Ubuntu

## Files and  Disk

Recursively coping files with specified extension**

```bash
cp --parents -R folder_src/**/*.mp4 ./folder_dst
```



**Batch rename files**

```bash
ls *jpg | awk -F\. '{print "mv "$0" C"NR".jpg"}' | sh
```



**Empty Trash in Command Line**

```bash
rm -rf ~/.local/share/Trash/*
```



**Calculate md5**

```bash
certutil -hashfile filename MD5 # windows
md5sum filename # ubuntu
```

**Summarize disk usage of each FILE recursively**

```bash
du -hs /path/to/directory # ubuntu
```

## Media

**Converting batches images into video**

```bash
ffmpeg -framerate 25 -i C%06d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
```



## Networks

**Set SSH password login**



Set password for user

```bash
sudo passwd user-name
```

Enable SSH password authentication

```bash
sudo vim /etc/ssh/sshd_config
# set PasswordAuthentication as yes
```

Restart SSH service

```bash
sudo service sshd restart
```

