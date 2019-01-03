---
layout: post
title:  Add shadowsocks to the service of Ubuntu
date:   2019-01-03 10:33:00 +0800
categories: [tools]
---

ref:
* https://blog.huihut.com/2017/08/25/LinuxInstallConfigShadowsocksClient/
* https://teddysun.com/486.html

开机自启
以下使用Systemd来实现shadowsocks开机自启。

sudo vim /etc/systemd/system/shadowsocks.service
在里面填写如下内容：

[Unit]
Description=Shadowsocks Client Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/sslocal -c /home/xx/Software/ShadowsocksConfig/shadowsocks.json

[Install]
WantedBy=multi-user.target
把/home/xx/Software/ShadowsocksConfig/shadowsocks.json修改为你的shadowsocks.json路径，如：/etc/shadowsocks.json

配置生效：

systemctl enable /etc/systemd/system/shadowsocks.service