- [AWS](#aws)
  - [1.Connect to linux](#1connect-to-linux)
  - [2.Configure scheduled tasks on remote linux](#2configure-scheduled-tasks-on-remote-linux)
  - [3.Use python on remote desktop](#3use-python-on-remote-desktop)
  - [4. Linux file exchange](#4-linux-file-exchange)

## AWS

The AWS folder contains all files within the remote AWS. This document is an introduction to using the cloud system.

#### 1.Connect to linux

After preparing harvey_linux.pem

```bash
ssh -i harvey_linux.pem ec2-user@
```

#### 2.Configure scheduled tasks on remote linux

Use the crontab command to set up scheduled tasks. Remember to enable scheduled task access permissions

```bash
chmod +x /home/ec2-user/quant/script.py
```

#### 3.Use python on remote desktop

Use python3 and pip3 for operations.

```bash
pip3 install tensorflow-cpu --no-cache-dir
```

#### 4. Linux file exchange

Copy only

```bash
scp -i harvey_linux.pem -r /Users/harvey/Desktop/MY_porject/quant3/AWS/LINUX ec2-user@:/home/ec2-user/quant
```

Maintain exact replica

```bash
rsync -avz --delete -e "ssh -i harvey_linux.pem" /Users/harvey/Desktop/quant2/AWS_remote_desktop/20240307/ ec2-user@:/home/ec2-user/quant/
```

Overwrite files with same name

```bash
rsync -avz -e "ssh -i harvey_linux.pem" /Users/harvey/Desktop/quant2/AWS_remote_desktop/20240307/ ec2-user@:/home/ec2-user/quant/
```

Overwrite files with same name but skip state

```bash
rsync -avz --exclude '/state/\*' -e "ssh -i harvey_linux.pem" /Users/harvey/Desktop/quant2/AWS_remote_desktop/20240319/ ec2-user@:/home/ec2-user/quant/
```

Download files to local

```bash
scp -i harvey_linux.pem -r ec2-user@:/home/ec2-user/quant/state /Users/harvey/Desktop/quant2/AWS_download
```
