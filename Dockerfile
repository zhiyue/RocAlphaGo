#基于tensorflow/tensorflow:latest-gpu的docker镜像构建
FROM tensorflow/tensorflow:latest-gpu
#暴露8888端口
EXPOSE 8999
#执行mkdir命令
RUN mkdir /app
#启动container后的默认路径为/app
WORKDIR /app
#将requirements.txt文件复制到镜像中的/app/requirements.txt
COPY requirements.txt /app/requirements.txt
#执行pip命令
RUN pip install -r requirements.txt
#将当前目录复制到镜像/app目录下
COPY . /app