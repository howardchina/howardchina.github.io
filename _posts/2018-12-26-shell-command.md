---
layout: post
title:  Shell command
date:   2018-12-26 23:49:00 +0800
categories: [shell]
---

* 用tar命令批量解压某个文件夹下所有的tar.gz文件

  ```shell
ls *.tar.gz | xargs -n1 tar xzvf
  ```

* mount *.ISO file

  ```shell
  sudo mount -o loop -t iso9660 ./Downloads/Linux/R2018a_glnxa64_dvd1.iso /media/mathworks/
  ```

