import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
aug = iaa.Rain(speed=(0.1, 0.3))

cap = cv2.VideoCapture("challenge.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)



out_dir = "challenge_clouds.mp4"
ret, frame = cap.read()
h, w = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
w_ = w + 2 * w // 4
h_ = h + 200
writer = cv2.VideoWriter(out_dir, fourcc, fps, (w, h))
display = 1
save = 1
images = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # frame = np.fliplr(frame)  # uncomment this line to flip the frame horizontally
        images.append(frame) 
    else:
        print("[INFO]: Video Finished!!!")
        break    

augmentation = iaa.Sequential([
    # iaa.Rain(speed=(0.1, 0.3)),
    # iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03)),
    iaa.Clouds()
])



augmented_images = augmentation(images=images)

for frame in augmented_images:
    print(frame)
    if display:
        cv2.imshow("frame", frame)
        k = cv2.waitKey(30)
        if k == ord('q') or k == 27:
            print("[WARN]: QUIT signal received. Exiting Process!!!")
            exit()
        elif k == ord('p'):
            print("[WARN]: PAUSE signal received. Halting Process. Press any key to continue")
            cv2.waitKey(0)

    if save:
        writer.write(frame)

if save:
    writer.release()    