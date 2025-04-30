import cv2
import mediapipe as mp
import tkinter as tk
import math
import threading
import pyautogui


screen_width, screen_height = pyautogui.size()  
class HandDetector:
    def __init__(self, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.handsModule = mp.solutions.hands
        self.hands = self.handsModule.Hands(
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )

    def get_landmarks(self, image):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imageRGB)
        lmList = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy, lm.z))
        return lmList

    def fingers_up(self, lmList):
        fingers = [0, 0, 0, 0, 0]
        if not lmList:
            return fingers
        fingers[0] = 1 if lmList[4][2] < lmList[2][2] else 0    #thumb
        fingers[1] = 1 if lmList[8][2] < lmList[6][2] else 0    # Index
        fingers[2] = 1 if lmList[12][2] < lmList[10][2] else 0  # Middle
        fingers[3] = 1 if lmList[16][2] < lmList[14][2] else 0  # Ring
        fingers[4] = 1 if lmList[20][2] < lmList[18][2] else 0  # Pinky
        return fingers

class SmartBoard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Drawing Board")
        self.width = 1000
        self.height = 800
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='white')
        self.canvas.pack()

        # Drawing properties
        self.color = "black"
        self.pen_size = 5
        self.eraser_size = 30
        self.prev_x, self.prev_y = None, None

        self.buttons = {
            "red": (20, 20, 120, 70),
            "green": (140, 20, 240, 70),
            "black": (260, 20, 360, 70)
        }

        self.draw_buttons()

    def draw_buttons(self):
        self.canvas.delete("buttons")
        for color, (x1, y1, x2, y2) in self.buttons.items():
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, tags="buttons")
            self.canvas.create_text((x1+x2)//2, (y1+y2)//2, text=color.capitalize(),
                                    fill="white" if color != "green" else "black", font=("Arial", 14), tags="buttons")

    def check_button_click(self, x, y):
        for color, (x1, y1, x2, y2) in self.buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.color = color
                print(f"Color changed to: {color}")
                return True
        return False

    def draw(self, x, y):
        if self.prev_x is not None and self.prev_y is not None:
            dist = math.hypot(x - self.prev_x, y - self.prev_y)
            if dist < 50:
                self.canvas.create_line(self.prev_x, self.prev_y, x, y,
                                        fill=self.color, width=self.pen_size, capstyle=tk.ROUND)
        self.prev_x, self.prev_y = x, y

    def show_eraser_preview(self, x, y):
        self.canvas.delete("eraser_preview")  
        self.canvas.create_oval(
            x - self.eraser_size, y - self.eraser_size,
            x + self.eraser_size, y + self.eraser_size,
            outline='gray', width=2, tags="eraser_preview"
        )


    def erase(self, x, y):
        self.canvas.create_oval(x - self.eraser_size, y - self.eraser_size,
                                x + self.eraser_size, y + self.eraser_size,
                                fill='white', outline='white')
        self.prev_x, self.prev_y = None, None

    def reset(self):
        self.prev_x, self.prev_y = None, None
        self.canvas.delete("eraser_preview")

    def run(self):
        self.root.mainloop()

def video_loop(board: SmartBoard):
    cap = cv2.VideoCapture(0)
    cap.set(3, board.width)
    cap.set(4, board.height)
    detector = HandDetector()

    drawing_enabled = True

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (board.width, board.height))

        lmList = detector.get_landmarks(img)
        if lmList:
            fingers = detector.fingers_up(lmList)
            ix, iy = lmList[8][1], lmList[8][2]  # Index tip
            thumb_x,thumb_y = lmList[4][1],lmList[4][2]

            diff_x = ix - thumb_x
            diff_y = iy - thumb_y

            distance = math.hypot(diff_x, diff_y)

            if distance < 30 and fingers == [1, 1, 0, 0, 0]:
                drawing_enabled = False
            else:
                drawing_enabled = True

            if drawing_enabled and fingers == [1, 1, 0, 0, 0]:
                if not board.check_button_click(ix, iy):
                    board.draw(ix, iy)
            elif fingers[1] and fingers[2] and  fingers[3] and not fingers[4]:
                board.erase(ix, iy)
                board.show_eraser_preview(ix, iy)
            else:
                board.reset()
        else:
            board.reset()

        # Show camera feed for debugging (optional)
        cv2.imshow("Camera Feed - Press 'q' to quit", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the app
if __name__ == "__main__":
    board = SmartBoard()
    threading.Thread(target=video_loop, args=(board,), daemon=True).start()
    board.run()
