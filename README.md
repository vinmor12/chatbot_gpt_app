# Chatbot GPT Application

OpenVINO application for text prediction with [GPT-2](https://github.com/vinmor12/chatbot_gpt_app/tree/main/model/gpt_2) and [GPT-Neo](https://github.com/vinmor12/chatbot_gpt_app/tree/main/model/gpt_neo) models.

It has been developed with Python 3.10 and it has been tested on Windows 11 with OpenVINO 2022.3.

Requirement
-
You can run this app with:
+ Python (version 3.7, 3.8, 3.9, 3.10)   
  see instruction [here](https://www.python.org/downloads/)
+ OpenVINO Toolkit (version 2022.3)  
  see instruction [here](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?VERSION=v_2023_2_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE)
+ Git (obviously)       
  see instruction [here](https://git-scm.com/)

Usage
-
You can use this application in the following way:
+ open a new Terminal and clone the repository
  ```
  git clone https://github.com/vinmor12/chatbot_gpt_app.git
  ```
+ activate OpenVINO environment  
  use the following command for Windows OS 
  ```
  openvino_env\Scripts\activate
  ```
  use the following command for Linux OS 
  ```
  source openvino_env/bin/activate
  ```
+ install the prerequisites if you don't have them  
  you need to install the following modules (if you don't have them)
  ```
  pip install gradio
  ```
  ```
  pip install ipywidgets
  ```
  ```
  pip install transformers
  ```
  ```
  pip install onnx
  ```
  ```
  pip install torch
  ```
+ switch to "chatbot_gpt_app" folder
  ```
  cd chatbot_gpt_app
  ```
+ run the application
  ```
  python chatbot_gpt.py
  ```
+ navigate the menu to download/convert available GPT models and to do infer on a supported target device.
<p align="center">
  <img width="35%" src="https://raw.githubusercontent.com/vinmor12/chatbot_gpt_app/main/data/images/main_menu_rev.png">
  <img width="35%" src="https://raw.githubusercontent.com/vinmor12/chatbot_gpt_app/main/data/images/inference_menu_rev.png">
</p>
<p align="center">
  <img width="70%" src="https://raw.githubusercontent.com/vinmor12/chatbot_gpt_app/main/data/images/gpt_2.png">
  <img width="70%" src="https://raw.githubusercontent.com/vinmor12/chatbot_gpt_app/main/data/images/gpt_neo.png">
</p>
