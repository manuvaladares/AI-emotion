import cv2
from deepface import DeepFace
import time

# ================= CONFIG ================= #

EMOTIONS = ["angry", "happy", "sad", "surprise", "neutral"]

# ---- Color palette  ---- #
WHITE = (235, 235, 235)
TEXT_GRAY = (200, 200, 200)
DARK_GRAY = (60, 60, 60)
MID_GRAY = (130, 130, 130)
ACCENT = (180, 130, 200)  

ANALYZE_EVERY = 0.6  # seconds

# ================= CAMERA ================= #

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

# ================= STATE ================= #

last_analysis_time = 0

emotion_scores = {e: 0 for e in EMOTIONS}
top_emotion = "neutral"
top_conf = 0

face_box = None  # (x, y, w, h)

# ================= FUNCTIONS ================= #

def analyze_emotion(frame_bgr):
    global emotion_scores, top_emotion, top_conf, face_box

    result = DeepFace.analyze(
        frame_bgr,
        actions=["emotion"],
        enforce_detection=False
    )

    data = result[0]

    # -------- Emotion scores -------- #
    emo_dict = data["emotion"]
    for e in EMOTIONS:
        emotion_scores[e] = float(emo_dict.get(e, 0.0))

    top_emotion = max(EMOTIONS, key=lambda e: emotion_scores[e])
    top_conf = int(emotion_scores[top_emotion])

    # -------- Face region -------- #
    if "region" in data:
        r = data["region"]
        if r["w"] > 0 and r["h"] > 0:
            face_box = (r["x"], r["y"], r["w"], r["h"])

# ================= MAIN LOOP ================= #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # -------- TOP HUD -------- #
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, "AI FACE EMOTION HUD",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                ACCENT, 2)

    # -------- ANALYSIS (timed) -------- #
    if time.time() - last_analysis_time > ANALYZE_EVERY:
        try:
            analyze_emotion(frame)
            last_analysis_time = time.time()
        except Exception:
            pass

    # -------- FACE BOX HUD -------- #
    if face_box:
        x, y, bw, bh = face_box

        padding = 20
        x = max(0, x - padding)
        y = max(40, y - padding)
        bw = min(w - x, bw + padding * 2)
        bh = min(h - y, bh + padding * 2)

        # Outer subtle border
        cv2.rectangle(frame,
                      (x - 2, y - 2),
                      (x + bw + 2, y + bh + 2),
                      MID_GRAY, 1)

        # Inner box
        cv2.rectangle(frame,
                      (x, y),
                      (x + bw, y + bh),
                      WHITE, 1)

        # Label
        label = f"{top_emotion} ({top_conf}%)"
        cv2.rectangle(frame,
                      (x, y - 24),
                      (x + bw, y),
                      (0, 0, 0), -1)

        cv2.putText(frame, label,
                    (x + 8, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    ACCENT, 2)

    # -------- SIDE PANEL -------- #
    panel_x = 20
    panel_y = 70
    line_h = 20
    max_bar_w = 110

    cv2.putText(frame, "emotions",
                (panel_x, panel_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                TEXT_GRAY, 1)

    for i, emo in enumerate(EMOTIONS):
        y_off = panel_y + i * line_h
        score = emotion_scores[emo]
        bar_w = int((score / 100.0) * max_bar_w)

        # emotion label
        cv2.putText(frame, emo,
                    (panel_x, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    TEXT_GRAY, 1)

        bar_y1 = y_off - 8
        bar_y2 = y_off - 2

        # bar background
        cv2.rectangle(frame,
                      (panel_x + 70, bar_y1),
                      (panel_x + 70 + max_bar_w, bar_y2),
                      DARK_GRAY, -1)

        # bar fill
        cv2.rectangle(frame,
                      (panel_x + 70, bar_y1),
                      (panel_x + 70 + bar_w, bar_y2),
                      ACCENT if emo == top_emotion else MID_GRAY,
                      -1)

    # -------- FOOTER -------- #
    cv2.putText(frame, "Press 'q' to quit",
                (w - 210, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                TEXT_GRAY, 1)

    cv2.imshow("AI Face Emotion HUD", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= CLEANUP ================= #

cap.release()
cv2.destroyAllWindows()
