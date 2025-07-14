from __future__ import annotations
import cv2, face_recognition, numpy as np, pickle, time, os, threading, queue, tkinter as tk, dlib
import pyttsx3
from cryptography.fernet import Fernet
import hashlib
from tkinter import simpledialog, messagebox  # import submodules for dialogs

# ───────────────────────── Config ──────────────────────────
DB_PATH           = "known_faces.enc"
KEY_PATH          = "db_key.key"
PASSWORD_HASH     = "5f4dcc3b5aa765d61d8327deb882cf99"  # md5("password")
THRESHOLD         = 0.45
GREETING_COOLDOWN = 10           # seconds
FRAME_SCALE       = 0.25         # was 0.17 → bump for better detection
SKIP_FRAMES       = 3            # heavy pass interval


USE_CUDA = False
DETECTION_MODEL = 'cnn' if USE_CUDA else 'hog'

# ──────────────────── Tkinter root setup ───────────────────
_root = tk.Tk()
_root.withdraw()

# ─────────────────── Access Control ────────────────────────
def ask_password():
    pwd = simpledialog.askstring("Login", "Enter password:", show="*")
    if pwd is None:
        exit(0)
    hashed = hashlib.md5(pwd.encode()).hexdigest()
    if hashed != PASSWORD_HASH:
        messagebox.showerror("Error", "Incorrect password.")
        exit(0)

ask_password()

# ─────────────────────── TTS helper ────────────────────────
class Speaker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.engine = pyttsx3.init()
        self.q: queue.Queue[str] = queue.Queue()
        self.start()
    def run(self):
        while True:
            txt = self.q.get()
            if txt is None:
                break
            self.engine.say(txt)
            self.engine.runAndWait()
    def speak(self, txt: str):
        self.q.put(txt)

audio = Speaker()

# ───────────────────────── DB utilities ───────────────────────

def load_key() -> bytes:
    if not os.path.exists(KEY_PATH):
        key = Fernet.generate_key()
        with open(KEY_PATH, 'wb') as f:
            f.write(key)
    else:
        with open(KEY_PATH, 'rb') as f:
            key = f.read()
    return key

fernet = Fernet(load_key())

def load_db() -> dict[str, list[np.ndarray]]:
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, 'rb') as f:
        encrypted = f.read()
    decrypted = fernet.decrypt(encrypted)
    return pickle.loads(decrypted)


def save_db(db: dict[str, list[np.ndarray]]):
    data = pickle.dumps(db)
    encrypted = fernet.encrypt(data)
    with open(DB_PATH, 'wb') as f:
        f.write(encrypted)

# ──────────────────── Name dialog prompt ───────────────────
def ask_name():
    win = tk.Toplevel(_root)
    win.title("New Face")
    win.attributes("-topmost", True)
    tk.Label(win, text="Enter a name for this person:").pack(padx=10, pady=(10, 0))
    name_var = tk.StringVar()
    entry = tk.Entry(win, textvariable=name_var)
    entry.pack(padx=10, pady=(0, 10))
    entry.focus_set()
    tk.Button(win, text="OK", command=win.destroy).pack(pady=(0, 10))
    win.grab_set()
    win.wait_window()
    return name_var.get().strip() or None

# ─────────────────── Camera worker thread ──────────────────
class Camera(threading.Thread):
    def __init__(self, index=0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam not found")
        self.frame: np.ndarray | None = None
        self.running = True
        self.start()
    def run(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                self.frame = f
    def read(self):
        return self.frame.copy() if self.frame is not None else None
    def stop(self):
        self.running = False
        self.cap.release()

# ───────────────────────── Main Logic ───────────────────────
def detect_with_fallback(rgb_small, small_factor):
    locs = face_recognition.face_locations(rgb_small, model=DETECTION_MODEL)
    if locs:
        return locs, small_factor
    for factor in (2, 4):
        img = cv2.resize(rgb_small, (0,0), fx=factor, fy=factor)
        locs = face_recognition.face_locations(img, model=DETECTION_MODEL)
        if locs:
            return [(t//factor, r//factor, b//factor, l//factor) for (t,r,b,l) in locs], small_factor/factor
    return [], small_factor


def main():
    known = load_db()
    greeted_at: dict[str, float] = {}
    cam = Camera()
    trackers: list[tuple[dlib.correlation_tracker, str]] = []
    tracker_enc: list[np.ndarray] = []
    frame_id = 0

    print("q → quit   |   s → save unknown face |   r → reset database")

    while True:
        frame = cam.read()
        if frame is None:
            cv2.waitKey(1)
            continue

        small = cv2.resize(frame, (0,0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        labels: list[str] = []
        locs: list[tuple[int,int,int,int]] = []

        heavy = (frame_id % SKIP_FRAMES == 0) or not trackers
        if heavy:
            locs, scale_used = detect_with_fallback(rgb_small, FRAME_SCALE)
            if scale_used == FRAME_SCALE:
                encs = face_recognition.face_encodings(rgb_small, locs)
            else:
                resized = cv2.cvtColor(
                    cv2.resize(frame, (0,0), fx=scale_used, fy=scale_used),
                    cv2.COLOR_BGR2RGB
                )
                scaled = [(int(t*scale_used/FRAME_SCALE), int(r*scale_used/FRAME_SCALE),
                           int(b*scale_used/FRAME_SCALE), int(l*scale_used/FRAME_SCALE)) for (t,r,b,l) in locs]
                encs = face_recognition.face_encodings(resized, scaled)

            trackers.clear(); tracker_enc.clear(); labels.clear()
            for (t,r,b,l), enc in zip(locs, encs):
                label, best = "Unknown", float('inf')
                for person, bank in known.items():
                    if bank:
                        d = np.min(face_recognition.face_distance(bank, enc))
                        if d < best:
                            label, best = person, d
                if best >= THRESHOLD:
                    label = "Unknown"

                st, sr, sb, sl = [int(v/FRAME_SCALE) for v in (t,r,b,l)]
                trk = dlib.correlation_tracker()
                trk.start_track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dlib.rectangle(sl,st,sr,sb))
                trackers.append((trk,label)); tracker_enc.append(enc); labels.append(label)

                if label != "Unknown" and time.time() - greeted_at.get(label, 0) > GREETING_COOLDOWN:
                    print(f"👋  Hello, {label}!"); audio.speak(f"Hello, {label}"); greeted_at[label] = time.time()
        else:
            for trk,label in trackers:
                trk.update(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pos = trk.get_position(); l,t,r,b = map(int,(pos.left(),pos.top(),pos.right(),pos.bottom()))
                locs.append((t,r,b,l)); labels.append(label)

        for (t,r,b,l), lbl in zip(locs, labels):
            cv2.rectangle(frame, (l,t),(r,b), (0,255,0), 2)
            cv2.rectangle(frame, (l,b-20),(r,b),(0,255,0), cv2.FILLED)
            cv2.putText(frame, lbl, (l+4,b-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1)

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('s'):
            if "Unknown" not in labels:
                print("⚠️  No *Unknown* face detected — press s when label reads 'Unknown'.")
            else:
                idx = labels.index("Unknown"); name = ask_name()
                if name:
                    known.setdefault(name, []).append(tracker_enc[idx]); save_db(known); print(f"✅  Saved {name}.")
        if key == ord('r'):
            known.clear();
            if os.path.exists(DB_PATH): os.remove(DB_PATH);
            print("🔄  Database cleared.")

        frame_id += 1

    cam.stop(); cv2.destroyAllWindows(); audio.speak(None)

if __name__ == "__main__":
    try: main()
    except Exception as e: print("Fatal error:", e)