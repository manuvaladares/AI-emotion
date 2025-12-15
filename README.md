# AI Face Emotion HUD

AI Face Emotion HUD is a Python-based application for real-time facial emotion recognition using  **[DeepFace](https://github.com/serengil/deepface)**, a lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for python. The system captures video from a webcam, detects a face, estimates emotional states, and overlays the results on the screen using a minimal HUD-style interface. The visual design prioritizes clarity and low visual impact, while the implementation focuses on robustness and performance by limiting how often emotion analysis is executed.

---

## Tracked Emotions

To keep the interface simple and readable, the system tracks a reduced set of emotions:

* angry
* happy
* sad
* surprise
* neutral

---

## Requirements

Install the required dependencies:

```bash
pip install opencv-python deepface
```

Note: DeepFace may download pre-trained models on first execution.

---

## How to Run

```bash
python emotion.py
```


## License

This code is an adaptation and extension of work originally shared by the GitHub user [SaniyaLadanavar16](https://github.com/SaniyaLadanavar16), with modifications to the visual style, interaction behavior, and overall structure. This project is provided for educational and experimental purposes. You are free to modify and adapt it as needed :)
