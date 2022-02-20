# sign-recog-api

This repository is dedicated to the api for the sign language recognition api.

Setup guide:

1. Create virtual env
   python3 -m venv venv
   . venv/bin/activate or ./venv/Scripts/activate on windows

2. Install packages:
   pip install -r requirements.txt

3. flask run --host=0.0.0.0 --port=5000 (the host is your ip, which you can get with ipconfig, while the port is your desired port), it is best to run flask run --host=0.0.0.0 --port=5000

4. To actually be able to send requests from the react native app to the flask api, you need to use localtunnel https://github.com/localtunnel/localtunnel
