import argparse
import sys

import cv2
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient


def build_parser():
    parser = argparse.ArgumentParser(description="Stream hand landmarks to TouchDesigner over OSC.")
    parser.add_argument("--ip", default="127.0.0.1", help="OSC destination IP address")
    parser.add_argument("--port", type=int, default=7000, help="OSC destination port")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--no-display", action="store_true", help="Hide the webcam window")
    return parser


def main():
    args = build_parser().parse_args()

    try:
        client = SimpleUDPClient(args.ip, args.port)
        hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}. Install it with: pip install opencv-python mediapipe python-osc")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}.")
        return 1

    print("--- Dual Hand Tracker Active ---")
    print(f"Sending OSC to {args.ip}:{args.port}")
    print("Press 'q' in the video window to stop.\n")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera. Exiting.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label.upper()

                    wrist = hand_landmarks.landmark[0]
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    middle_tip = hand_landmarks.landmark[12]

                    client.send_message(f"/{hand_label.lower()}/wrist/x", wrist.x)
                    client.send_message(f"/{hand_label.lower()}/wrist/y", wrist.y)
                    client.send_message(f"/{hand_label.lower()}/thumb/x", thumb_tip.x)
                    client.send_message(f"/{hand_label.lower()}/thumb/y", thumb_tip.y)
                    client.send_message(f"/{hand_label.lower()}/index/x", index_tip.x)
                    client.send_message(f"/{hand_label.lower()}/index/y", index_tip.y)
                    client.send_message(f"/{hand_label.lower()}/middle/x", middle_tip.x)
                    client.send_message(f"/{hand_label.lower()}/middle/y", middle_tip.y)

            if not args.no_display:
                cv2.imshow("MediaPipe Hand Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
