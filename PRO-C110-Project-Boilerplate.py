import cv2
import numpy as np
import tensorflow as tf

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Load the Keras model
model_path = "/Users/gautammahesh/Downloads/PRO-C110-Project-Boilerplate-main/keras_model.h5"
model = tf.keras.models.load_model(model_path)

# Infinite loop
while True:
    # Reading / Requesting a Frame from the Camera 
    status, frame = camera.read()

    # if we were successfully able to read the frame
    if status:
        # Flip the frame
        frame = cv2.flip(frame, 1)

        # resize the frame
        frame_resized = cv2.resize(frame, (224, 224))

        # expand the dimensions
        input_frame = np.expand_dims(frame_resized, axis=0)

        # normalize it before feeding to the model
        input_frame = input_frame / 255.0

        # Perform prediction
        predictions = model.predict(input_frame)

        # displaying the frames captured
        cv2.imshow('feed', frame)

        # waiting for 1ms
        code = cv2.waitKey(1)

        # if space key is pressed, break the loop
        if code == 32:
            break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
