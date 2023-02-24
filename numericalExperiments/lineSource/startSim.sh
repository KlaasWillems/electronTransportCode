#!/bin/bash
# Run command and leave
# Source: https://askubuntu.com/questions/1143698/how-to-run-ssh-on-remote-machine-and-leave
rm data/logfile.txt
/bin/python3 -O /home/r0738465/Documents/electronTransportCode/numericalExperiments/lineSource/LineSourceScript.py >> data/logfile.txt &
disown
# tail -f data/logfile.txt
