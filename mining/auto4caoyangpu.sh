#!/bin/sh
isrunning=0
processid=0
dayflag=0
nightflag=0
while [ 1 ]
do
hour=$(date | cut -d ' ' -f 5 | cut -d ':' -f 1)
minute=$(date | cut -d ' ' -f 5 | cut -d ':' -f 2)
second=$(date | cut -d ' ' -f 5 | cut -d ':' -f 3)
pidof "./testCNN"
if [ $? -eq 0 ]; then
	isrunning=1
	processid=$(pidof "./testCNN")
else
	isrunning=0
	processid=0
fi
if [ $hour -ge 9 ] && [ $hour -le 23 ] && [ $isrunning -eq 0 ]; then
	cd /data/caoyangpu/javen/dataset/ethminer-master/build/ethminer
	nohup ./testCNN --farm-recheck 200 -U --cuda-devices 5 6 7 -F http://10.212.45.119:8098/caoyangpu &
	isrunning=1
	dayflag=1
	nightflag=0
elif [ $hour -ge 9 ] && [ $hour -le 23 ] && [ $isrunning -eq 1 ] && [ $nightflag -eq 1 ]; then
	kill $processid
	nightflag=0
	cd /data/caoyangpu/javen/dataset/ethminer-master/build/ethminer
	nohup ./testCNN --farm-recheck 200 -U --cuda-devices 5 6 7 -F http://10.212.45.119:8098/caoyangpu &
	dayflag=1
	isrunning=1
elif [ $hour -lt 9 ] || [ $hour -gt 23 ] && [ $isrunning -eq 0 ]; then
	cd /data/caoyangpu/javen/dataset/ethminer-master/build/ethminer
	nohup ./testCNN --farm-recheck 200 -G -F http://10.212.45.119:8088/mountain &
	nightflag=1
	dayflag=0
	isrunning=1
elif [ $hour -lt 9 ] || [ $hour -gt 23 ] && [ $isrunning -eq 1 ] && [ $dayflag -eq 1 ]; then
	kill $processid
	dayflag=0
	cd /data/caoyangpu/javen/dataset/ethminer-master/build/ethminer
	nohup ./testCNN --farm-recheck 200 -G -F http://10.212.45.119:8088/mountain &
	nightflag=1
	isrunning=1
fi
done
