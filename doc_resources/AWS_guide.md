- [AWS](#aws)
  - [1.连接 linux](#1连接-linux)
  - [2.配置远程 linux 的定时任务](#2配置远程-linux-的定时任务)
  - [3.在远程桌面使用 python](#3在远程桌面使用-python)
  - [4. linux 的文件交换](#4-linux-的文件交换)

## AWS

AWS 文件夹下是远程 AWS 内的所有文件。该文档是关于使用云系统的简介。

#### 1.连接 linux

准备好 harvey_linux.pem 后

```bash
ssh -i harvey_linux.pem ec2-user@
```

#### 2.配置远程 linux 的定时任务

使用 crontab 命令设置定时任务。记得要开启定时任务访问权限

```bash
chmod +x /home/ec2-user/quant/script.py
```

#### 3.在远程桌面使用 python

使用 python3 和 pip3 来操作。

```bash
pip3 install tensorflow-cpu --no-cache-dir
```

#### 4. linux 的文件交换

仅拷贝

```bash
scp -i harvey_linux.pem -r /Users/harvey/Desktop/MY_porject/quant3/AWS/LINUX ec2-user@:/home/ec2-user/quant
```

保持精确的副本

```bash
rsync -avz --delete -e "ssh -i harvey_linux.pem" /Users/harvey/Desktop/quant2/AWS_remote_desktop/20240307/ ec2-user@:/home/ec2-user/quant/
```

覆盖同名文件

```bash
rsync -avz -e "ssh -i harvey_linux.pem" /Users/harvey/Desktop/quant2/AWS_remote_desktop/20240307/ ec2-user@:/home/ec2-user/quant/
```

覆盖同名文件但是跳过 state

```bash
rsync -avz --exclude '/state/\*' -e "ssh -i harvey_linux.pem" /Users/harvey/Desktop/quant2/AWS_remote_desktop/20240319/ ec2-user@:/home/ec2-user/quant/
```

下载文件到本地

```bash
scp -i harvey_linux.pem -r ec2-user@:/home/ec2-user/quant/state /Users/harvey/Desktop/quant2/AWS_download
```
