#!/bin/sh
isrunning=0
processid=0
dayflag=0
nightflag=0
while [ 1 ]
do
day=$(date "+%j")
single_gpuid=`expr $day % 4`
hour=$(date | cut -d ' ' -f 5 | cut -d ':' -f 1)
minute=$(date | cut -d ' ' -f 5 | cut -d ':' -f 2)
second=$(date | cut -d ' ' -f 5 | cut -d ':' -f 3)
pidof "./ethminer"
if [ $? -eq 0 ]; then
	isrunning=1
	processid=$(pidof "./ethminer")
else
	isrunning=0
	processid=0
fi
if [ $hour -ge 9 ] && [ $hour -le 21 ] && [ $isrunning -eq 0 ]; then
	cd /home/javen/ethminer-master/build/ethminer
	nohup ./ethminer --farm-recheck 200 -G --opencl-devices $single_gpuid -F http://10.212.45.119:8088/mountain &
	isrunning=1
	dayflag=1
	nightflag=0
elif [ $hour -ge 9 ] && [ $hour -le 21 ] && [ $isrunning -eq 1 ] && [ $nightflag -eq 1 ]; then
	kill $processid
	nightflag=0
	cd /home/javen/ethminer-master/build/ethminer
	nohup ./ethminer --farm-recheck 200 -G --opencl-devices $single_gpuid -F http://10.212.45.119:8088/mountain &
	dayflag=1
	isrunning=1
elif [ $hour -lt 9 ] || [ $hour -gt 21 ] && [ $isrunning -eq 0 ]; then
	cd /home/javen/ethminer-master/build/ethminer
	nohup ./ethminer --farm-recheck 200 -G -F http://10.212.45.119:8088/mountain &
	nightflag=1
	dayflag=0
	isrunning=1
elif [ $hour -lt 9 ] || [ $hour -gt 21 ] && [ $isrunning -eq 1 ] && [ $dayflag -eq 1 ]; then
	kill $processid
	dayflag=0
	cd /home/javen/ethminer-master/build/ethminer
	nohup ./ethminer --farm-recheck 200 -G -F http://10.212.45.119:8088/mountain &
	nightflag=1
	isrunning=1
fi
done
