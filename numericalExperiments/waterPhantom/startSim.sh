#!/bin/bash
# Run command and leave
# Source: https://askubuntu.com/questions/1143698/how-to-run-ssh-on-remote-machine-and-leave
rm data/logfile.txt
/bin/python3 /home/r0738465/Documents/electronTransportCode/numericalExperiments/waterPhantom/WaterPhantomScript.py >> data/logfile.txt &
tail -f data/logfile.txt
disown
