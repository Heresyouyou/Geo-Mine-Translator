@echo off
cd /d D:\PostNews\Translate

timeout /t 8 /nobreak >nul

start /min "" py -3.12 web_app.py
