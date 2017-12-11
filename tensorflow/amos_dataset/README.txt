- If you want to download the geolocation dataset, then run:

	python download_amos.py

  By default, this will download images from the geolocation dataset
  to the current working directory.  To change this behavior, change
  the definition of ROOT_LOCATION in download_amos.py.

  This file has several other constants that can be modified to
  download other large chunks of the AMOS dataset at a time.

  This will download and extract a directory of folders, indexed by
  camera ID and timestamp and organized by month.  Images are
  timestamped as yyyymmdd_hhmmss.jpg, in 24-hour UTC time.  For example, the
  image from camera 123 taken on April 5, 2008 at 6:12:34AM UTC time
  will be located at
  
      (current directory)/AMOS_Data/00000123/2008.04/20080405_061234.jpg



- If you want to download a month's worth of images, go to:

	http://amosweb.cse.wustl.edu/zipfiles/

  This directory tree has a listing of all images in the AMOS
  dataset.  The files are indexed by numeric camera ID and month
  as follows:
  
        /zipfiles/yyyy/<last two digits>/<last four digits>/<all eight digits>/yyyy.mm.zip

  for example, the images taken by camera 12345 in January 2011 are
  under  /zipfiles/2011/45/2345/00012345/2011.01.zip



- If you want to learn how to use the enclosed AMOSdataset.mat file read on:
  
  This .mat file is a (1x633) struct with three fields:
  (1) camid : the AMOS id for the camera
  (2) time : the set of timestamps (GMT, serial date number form) corresponding
             to when images were captured from the camera
  (3) avgI : the set of average image intensities corresponding to the images
             captured from the camera
  When loaded into Matlab, it will appear in your workspace as
  camera_time_light_data. You can access a camera's data as follows:
  
        camera_time_light_data(<index>).<field>
        
  for example, the AMOS id, the timestamps and the average image
  intensities for the fifth camera are accessed through
  camera_time_light_data(5).camid, camera_time_light_data(5).time,
  and camera_time_light_data(5).avgI, respectively.
  

  
- A note on the ground truth data:
  
  The ground truth data for the 633 cameras can be
  found in the AMOSgroundtruth.txt file. Each line of
  the file corresponds to the ground truth location
  for one camera and is formatted as follows:
  
  <camid> <latitude> <longitude>
  
  for example, the line
  
  1000 38.8921010100000 -77.0449380300000
  
  places the ground truth location of camera 1000
  at 38.8921010100000 lat, -77.0449380300000 long.
