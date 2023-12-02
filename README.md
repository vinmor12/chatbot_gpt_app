# Chatbot GPT Application

OpenVINO application to text prediction with **GPT-2** and **GPT-Neo** models.

It has been powered with Python 3.10 and it has been tested on Windows 11 with OpenVINO 2022.3.

Prerequisites
-
You can run this app with:
+ Python (version 3.7, 3.8, 3.9, 3.10)   
  see instruction [here](https://www.python.org/downloads/)
+ OpenVINO Toolkit (version 2022.3)  
  see instruction [here](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?VERSION=v_2023_2_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE)

In addition, you must install Gradio for Interactive Inference with the following command:
```
pip install gradio
```

Usage
-
You can use this application in the following way:
+ open a new Terminal and clone the repository
  ```
  git clone https://github.com/vinmor12/classification_app.git
  ```
+ activate OpenVINO environment (the following command is for Windows OS)
  ```
  openvino_env\Scripts\activate
  ```
+ switch to "chatbot_gpt_app" folder
  ```
  cd chatbot_gpt_app
  ```
+ run the application
  ```
  python chatbot_gpt.py
  ```
+ navigate the menu to download/convert aivables gpt models ad to do infer on a supported device target