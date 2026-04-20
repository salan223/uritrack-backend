#!/bin/bash
cd /home/eng4kteam09/kidney_stone_detector
exec /home/eng4kteam09/kidney_stone_detector/venv/bin/python -c "from communication.wifi_api import run_server; run_server()"
