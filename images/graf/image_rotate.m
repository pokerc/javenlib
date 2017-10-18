clear all;
a = imread('img1.ppm');
%imshow(a);
b = imrotate(a,90,'bilinear','crop');
%imshow(b);
imwrite(b,'img1_rotate90.ppm');