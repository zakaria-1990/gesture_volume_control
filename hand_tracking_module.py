import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, max_hands=2, complexity=0, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_con = detection_con
        self.track_con = track_con
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode, max_num_hands=self.max_hands,
                                         model_complexity=self.complexity,
                                         min_detection_confidence=self.detection_con,
                                         min_tracking_confidence=self.track_con)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)

        if self.results.multi_hand_landmarks:
            for landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, landmark, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for idx, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([idx, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return landmark_list


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while cap.isOpened():
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        cv2.imshow('img detector', img)
        if cv2.waitKey(5) & 0XFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
