import cv2
import numpy as np
from keras.models import model_from_json
import moviepy.editor as mp

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def get_avg_emotions_for_timings(video_path, timings):
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    emotion_model.load_weights("model/emotion_model.h5")
    print("Loaded model from disk")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    interval_frames = int(fps * 2)  # Number of frames in 2 seconds
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    avg_emotions = []

    for timing in timings:
        timing_frames = int(timing * fps)

        # Move to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, timing_frames)

        emotions_in_interval = []

        # Read frames until the end of the interval or end of video
        for _ in range(interval_frames):
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in num_faces:
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                emotions_in_interval.append(emotion_dict[maxindex])

        # Calculate average emotion for the interval
        if len(emotions_in_interval) > 0:
            avg_emotion = max(set(emotions_in_interval), key=emotions_in_interval.count)
        else:
            # print("--------------------------------------------------")
            avg_emotion = "Neutral"  # If no face detected, consider as Neutral emotion
        avg_emotions.append(avg_emotion)

    cap.release()
    cv2.destroyAllWindows()

    return avg_emotions

# Example usage:
video_path = "Vidinsta_Instagram Post_66169582d2a28.mp4"
end_timings=[1.151, 2.754, 5.018, 8.265, 12.432, 16.028, 20.552, 22.954, 26.857, 30.12, 33.062,36.425,48.32]
emotions_list = get_avg_emotions_for_timings(video_path, end_timings) 
print("Average emotions for provided timings:", emotions_list)

print(len(end_timings), len(emotions_list))