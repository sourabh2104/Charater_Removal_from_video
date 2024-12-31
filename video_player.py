import cv2

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow('Video', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Use the player
video_path = "/home/sourabh/video_remover/output/removed_person_Express_train_to_lower_manhattan.mp4"  # Replace with your video path
play_video(video_path)