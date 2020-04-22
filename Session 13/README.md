Convert images to video

ffmpeg -start_number 133 -i image-%03d.jpg woody.mp4

Convert video to image

ffmpeg -i video.mp4 image-%03d.jpg
