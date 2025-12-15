import cv2
from deepface import DeepFace
import time

# 5 emotions we care about
EMOTIONS = ["angry", "happy", "sad", "surprise", "neutral"]

# Colors
WHITE = (255, 255, 255)
NEON_PINK = (255, 0, 255)

# Analyze every N seconds for smooth performance
ANALYZE_EVERY = 0.6  # seconds

# Try different camera indices
cap = None
for index in [0, 1, 2]:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera opened successfully at index {index}")
        break
    cap.release()

if not cap or not cap.isOpened():
    print("Error: Could not open any camera")
    exit(1)

last_analysis_time = 0
emotion_scores = {e: 0 for e in EMOTIONS}
top_emotion = "neutral"
top_conf = 0


def analyze_emotion(frame_bgr):
    """Run DeepFace once and return scores for our 5 emotions."""
    global emotion_scores, top_emotion, top_conf

    result = DeepFace.analyze(
        frame_bgr,
        actions=["emotion"],
        enforce_detection=False
    )

    emo_dict = result[0]["emotion"]  # DeepFace full dict

    # Keep only our 5 emotions
    for e in EMOTIONS:
        emotion_scores[e] = float(emo_dict.get(e, 0.0))

    # Find top emotion among the 5
    top_emotion = max(EMOTIONS, key=lambda e: emotion_scores[e])
    top_conf = int(emotion_scores[top_emotion])


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Top HUD bar
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, "AI FACE EMOTION HUD",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, NEON_PINK, 2)

    # Run analysis every ANALYZE_EVERY seconds (smooth + faster)
    if time.time() - last_analysis_time > ANALYZE_EVERY:
        try:
            analyze_emotion(frame)
            last_analysis_time = time.time()
        except Exception:
            # If DeepFace fails for a frame, just keep previous values
            pass

    # ----- White face box (center HUD style) -----
    box_w, box_h = 260, 300
    x = w // 2 - box_w // 2
    y = h // 2 - box_h // 2

    # Outer glow-ish white box
    cv2.rectangle(frame, (x - 3, y - 3), (x + box_w + 3, y + box_h + 3),
                  (200, 200, 200), 2)
    # Inner solid white box
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), WHITE, 2)

    # Top label on box: main emotion + %
    label = f"{top_emotion} ({top_conf}%)"
    cv2.rectangle(frame, (x, y - 30), (x + box_w, y), (0, 0, 0), -1)
    cv2.putText(frame, label,
                (x + 10, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # ----- Side panel with all 5 emotions + bars -----
    panel_x = 20
    panel_y = 60
    line_h = 28
    max_bar_w = 160

    cv2.putText(frame, "Tracked emotions:",
                (panel_x, panel_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    for i, emo in enumerate(EMOTIONS):
        y_off = panel_y + i * line_h
        score = emotion_scores[emo]
        bar_w = int((score / 100.0) * max_bar_w)

        # emotion name
        cv2.putText(frame, f"{emo:8s}",
                    (panel_x, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        # bar background
        cv2.rectangle(frame,
                      (panel_x + 90, y_off - 12),
                      (panel_x + 90 + max_bar_w, y_off + 4),
                      (50, 50, 50), -1)

        # bar fill
        cv2.rectangle(frame,
                      (panel_x + 90, y_off - 12),
                      (panel_x + 90 + bar_w, y_off + 4),
                      NEON_PINK if emo == top_emotion else (180, 180, 180),
                      -1)

    cv2.putText(frame, "Press 'q' to quit",
                (w - 210, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    cv2.imshow(" AI Face Emotion HUD", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()