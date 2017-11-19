#!/usr/bin/python
#encoding=utf-8 
import os
def rename():
	count = 0
	name = 'cat'
	path = '/home/javen/caffe-master/data/myself/test/cat'
	filelist = os.listdir(path)
	for files in filelist:
		olddir = os.path.join(path,files)
		if os.path.isdir(olddir):
			continue
		filename = os.path.splitext(files)[0]
		filetype = os.path.splitext(files)[1]
		newdir = os.path.join(path,name+str(count)+filetype)
		os.rename(olddir,newdir)
		count += 1

rename()
