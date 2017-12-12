# a helper utility to download and unzip a lot of images from the AMOS dataset.

import os
import sys
import urllib2
import StringIO
import zipfile

# Change this to where you want data to be dumped off.  If not supplied, defaults to
# the current working directory.
# example:
# ROOT_LOCATION = '/path/to/where/you/want/AMOS_Data/'
ROOT_LOCATION = './'

# change these parameters as necessary to download whichever camera or year or month you
# want to download. These initial, 633 cameras are the cameras we used in our WACV 2011
# paper - you may want to keep them around if you are planning on comparing your results
# directly to our algorithm.
# CAMERAS_TO_DOWNLOAD = [63, 64, 65, 76, 78, 80, 82, 83, 84, 87, 89, 90, 93, 95, 163, 167, 168, 169, 170, 171, 174, 192, 194, 196, 198, 199, 200, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 222, 223, 230, 231, 232, 234, 235, 237, 242, 246, 247, 248, 249, 250, 251, 252, 254, 255, 256, 257, 258, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 273, 276, 277, 278, 279, 281, 282, 284, 285, 286, 287, 288, 289, 290, 291, 292, 295, 297, 298, 299, 300, 301, 302, 303, 305, 306, 307, 311, 312, 313, 314, 315, 316, 319, 321, 322, 323, 324, 325, 327, 328, 329, 332, 333, 334, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 361, 362, 364, 366, 367, 368, 370, 371, 372, 373, 374, 375, 376, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 391, 392, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 409, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 426, 427, 428, 431, 432, 433, 434, 435, 437, 438, 440, 441, 443, 444, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 461, 462, 463, 464, 465, 466, 467, 470, 471, 472, 473, 476, 477, 478, 480, 481, 484, 487, 488, 489, 490, 491, 492, 494, 495, 496, 497, 498, 499, 501, 502, 503, 504, 505, 506, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 524, 527, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 564, 565, 567, 568, 569, 571, 572, 574, 575, 576, 578, 579, 580, 581, 582, 583, 584, 585, 587, 588, 589, 590, 592, 593, 594, 597, 600, 602, 604, 606, 607, 608, 609, 610, 611, 612, 613, 616, 618, 619, 620, 623, 625, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 646, 648, 650, 652, 653, 654, 655, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 669, 670, 671, 673, 674, 675, 676, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 690, 691, 692, 693, 694, 695, 696, 697, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 713, 715, 716, 717, 718, 720, 721, 723, 724, 725, 726, 730, 732, 733, 734, 735, 736, 763, 790, 797, 798, 800, 801, 802, 803, 806, 807, 808, 809, 810, 812, 814, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 832, 833, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 850, 851, 852, 854, 855, 856, 857, 858, 859, 864, 870, 871, 872, 873, 878, 879, 881, 885, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 907, 909, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021]
CAMERAS_TO_DOWNLOAD = [63]
YEARS_TO_DOWNLOAD = [2008]
MONTHS_TO_DOWNLOAD = range(1,13)

# if the script crashed or the power went out or something, this flag will
# skip downloading and unzipping a month's worth of images if there's already
# a folder where it should be.  If you set this to false, then downloads
# will overwrite any existing files in case of filename conflict.
SKIP_ALREADY_DOWNLOADED = True

# don't change this.
ZIPFILE_URL = 'http://amosweb.cse.wustl.edu/zipfiles/'


def download(camera_id, month, year):
    """
    Downloads a zip file from AMOS, returns a file.
    """
    last_two_digits = camera_id % 100;
    last_four_digits = camera_id % 10000;
    
    url = ZIPFILE_URL + '%04d/%02d/%04d/%08d/%04d.%02d.zip' % (year, last_two_digits, last_four_digits, camera_id, year, month)
    print '    downloading...',
    sys.stdout.flush()
    
    try:
        result = urllib2.urlopen(url)
    except urllib2.HTTPError as e:
        print e.code, 'error.'
        
    handle = StringIO.StringIO(result.read())
    
    print 'done.'
    sys.stdout.flush()
    
    return handle
    
def extract(file_obj, location):
    """
    Extracts a bunch of images from a zip file.
    """
    print '    extracting zip...',
    sys.stdout.flush()
    
    zf = zipfile.ZipFile(file_obj, 'r')
    zf.extractall(location)
    zf.close()
    file_obj.close()
    
    print 'done.'
    sys.stdout.flush()
    
def ensure_directory_exists(path):
    """
    Makes a directory, if it doesn't already exist.
    """
    dir_path = path.rstrip('/')       
 
    if not os.path.exists(dir_path):
        parent_dir_path = os.path.dirname(dir_path)
        ensure_directory_exists(parent_dir_path)

        try:
            os.mkdir(dir_path)
        except OSError:
            pass
			
			

def main():
    # for all cameras...
    for camera_id in CAMERAS_TO_DOWNLOAD:
        # for all years...
        for year in YEARS_TO_DOWNLOAD:
            # for all months of imagery...
            for month in MONTHS_TO_DOWNLOAD:
                
                location = ROOT_LOCATION + '%08d/%04d.%02d/' % (camera_id, year, month)
                print location
                
                if SKIP_ALREADY_DOWNLOADED and os.path.exists(location):
                    print "     already downloaded."
                    continue

                # download the month.
                zf = download(camera_id, month, year)

                # extract the month.
                ensure_directory_exists(location)
                extract(zf, location)
            
if __name__ == '__main__':
    
    if ROOT_LOCATION == None:
        ROOT_LOCATION = os.getcwd() + '/AMOS_Data'
        
    if ROOT_LOCATION[-1] != '/':
        ROOT_LOCATION = ROOT_LOCATION + '/'
    print 'Downloading images to:'
    
    main()
