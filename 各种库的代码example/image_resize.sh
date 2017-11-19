#!/bin/sh

for name in /home/javen/caffe-master/data/myself/train/cat/*.jpg;do
	echo $name
	convert -resize 256x256\! $name $name
done
