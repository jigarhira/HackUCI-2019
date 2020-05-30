# HackUCI-2019 [Winner: Best use of Embedded Systems]
# DriveAlert
https://devpost.com/software/drivealert

DriveAlert is a way to extend smart-car safety to everyone. As you drive, it takes note of how frequently you close your eyes and alerts you when you've surpassed the threshold, which is a likely indicator you've begun to fall asleep. DriveAlert will then play music to wake you back up and also send a text message to your designated emergency contacts notifying them that you've fallen asleep at the wheel and telling them to come pick you up.
#
![DriveAlert Device](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/000/766/598/datas/gallery.jpg)
#
## How it Works
We used OpenCV to determine whether the user's eyes were present and if they were, whether they were open or not. We used Tensorflow and Keras to process the eyes through a model trained by a CNN. Then, we used Flask to build an API in which we exposed this data for consumption by our frontend, which displays the current stats. For music, we used aplay to play music through a connected bluetooth device. And for text notifications, we used the Twilio SMS API.
#
![DriveAlert Hardware](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/000/767/016/datas/gallery.jpg)
