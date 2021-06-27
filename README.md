 ![](https://img.shields.io/badge/microsoft%20azure-0089D6?style=for-the-badge&logo=microsoft-azure&logoColor=white) ![](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white) ![](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white) ![](https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white) ![](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) ![](https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E) ![](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
 
# **Deploy TensorFlow 2 Object Detection API models with Python-Flask**

This website allows you to deploy the models you created with TensorFlow 2 Object Detection API on the server and make them available to other people. Thus, the models you write will not stay in the air and people will be able to use them. The main purpose of artificial intelligence is to solve problems, so you will be able to solve problems in daily life.


### How the application works ?

The operation of the application is given in the diagram below, the image taken from the client, ie people, reaches the server, and these images are processed here, then the final state of the image array is visualized on the front with html.

![asd](https://user-images.githubusercontent.com/54184905/123537436-1e371980-d738-11eb-8d5b-6cb35ed355a0.png)


### Steps to run the web application on the server

First of all, install the python packages in the requirements.txt to the server (you can install these packages with the command below).

```console
pip3 install -r requirements.txt
```

After installing the Python packages, we need to make the necessary tensorflow package configurations (just run the commands below).

```console
cd tf
./bin/protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`/:`pwd`/slim
cd ..
```

After installing the necessary python packages and tensorflow configuration, we must make the https and 443 ports public on the server (Run the following commands in order).

```console
sudo systemctl status ufw 
sudo ufw enable
sudo ufw allow https
sudo ufw allow 443
```

After completing these processes, you can now access the website with public ip, but we still have some shortcomings, we must create an SSL certificate and include it in python flask, otherwise your browser will not trust the website and will not open your web cam (run the following commands).

```console
pip3 install pyopenssl
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```

It will create the necessary certificates in the folder and I have included these certificates in the code for you, you don't need to do anything.

```python
socketio.run(app, host="0.0.0.0", port=443, ssl_context=('cert.pem', 'key.pem'))
```

Everything is ok now, your python site will be up with the following command and you can access the site by going to https://xxxx. I will go to https://168.62.57.147, you can edit the url part according to you.

```console
python3 main.py
```

**Information :** In the current application, we will use the model named "ssd_mobilenet_v2_320x320_coco17_tpu-8" and we will use ready-made "coco" weights and classes, if you wish, you can use this web application with your own model and your own classes.

### Test image and video

[v](https://user-images.githubusercontent.com/54184905/123538349-234a9780-d73d-11eb-9334-f811bfdf3822.mp4)

![Screenshot_20210626-101819_Chrome](https://user-images.githubusercontent.com/54184905/123538664-b2a47a80-d73e-11eb-8cff-f384cd52fe85.jpg)

