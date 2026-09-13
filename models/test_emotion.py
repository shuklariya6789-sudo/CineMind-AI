import cv2

try:
    from emotion import detect_emotion
except FileNotFoundError as e:
    print(f"Could not load emotion models:\n{e}")
    exit(1)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open nahi ho raha!")
    exit(1)

print("Camera started. Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame nahi mil raha!")
            break

        results = detect_emotion(frame)

        for result in results:

            x1, y1, x2, y2 = result["box"]
            emotion = result["emotion"]
            probabilities = result["probabilities"]

            # Face box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Detected emotion
            cv2.putText(
                frame,
                emotion.upper(),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Show probabilities
            y_position = 30

            for name, probability in probabilities.items():

                text = f"{name}: {probability:.1f}%"

                cv2.putText(
                    frame,
                    text,
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )

                y_position += 25

        cv2.imshow(
            "AI Emotion Detection - Probabilities",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Always release the camera and close windows, even if something
    # above raises (e.g. a bad frame or a display error).
    cap.release()
    cv2.destroyAllWindows()