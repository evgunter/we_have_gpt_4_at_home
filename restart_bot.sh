#!/bin/bash

LOGFILE="cron.log"

echo "$(date +%s) Killing any running bot_script.py instances..." >> $LOGFILE
pkill -f "python3 bot_script.py" && echo "$(date) Successfully killed bot_script.py" >> $LOGFILE || echo "$(date) bot_script.py not running, skipping kill" >> $LOGFILE

echo "$(date +%s) Restarting bot_script.py" >> $LOGFILE
nohup python3 bot_script.py >/dev/null 2>>$LOGFILE &
