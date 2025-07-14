📄 README.md
markdown
Copy code
# Face Recognition Attendance System

A real-time face recognition application built with OpenCV, `face_recognition`, `dlib`, and Tkinter GUI. This tool identifies faces from a webcam feed, greets known individuals via text-to-speech, and allows dynamic updates to the encrypted face database.

---

## 🚀 Features

- 🔒 **Password-Protected Access**: Launch requires an authorized password.
- 🧠 **Real-Time Face Recognition**: Detects and identifies known faces using a live webcam feed.
- 🔐 **Encrypted Face Database**: Face encodings are securely stored with AES encryption via Fernet.
- 🎙️ **Text-to-Speech Greetings**: Greets recognized users via `pyttsx3` TTS engine.
- ➕ **Add New Faces on the Fly**: Press `s` when an unknown face is detected to label and save it.
- 🔄 **Reset Database**: Press `r` to clear all stored identities.

---

## 📷 Usage Instructions

1. **Run the script**:

   ```bash
   python main.py
   ```
2. **At startup:**
    A GUI prompt will ask for the password (default hash is for "password" — MD5 hashed).
3. **Keyboard Shortcuts:**

- `q` – Quit the application  
- `s` – Save a new unknown face (GUI prompt for name)  
- `r` – Reset/clear the entire face database



## 📦 Requirements
**Install all dependencies via pip:**
   ```
pip install -r requirements.txt
```


Dependencies include:

- `face-recognition`
- `dlib`
- `opencv-python`
- `pyttsx3`
- `cryptography`
- `tkinter` (usually bundled with Python)
- See `requirements.txt` for full list.
## 💾 File Structure

| File / Folder         | Description                                      |
|-----------------------|--------------------------------------------------|
| `main.py`             | Main application script                          |
| `requirements.txt`    | Python dependency list                           |
| `FaceRecognition.spec`| PyInstaller build spec (optional)                |
| `known_faces.enc`     | Encrypted face encoding database (ignored in Git)|
| `db_key.key`          | Fernet encryption key (ignored in Git)           |
| `.gitignore`          | Prevents large/sensitive files from uploading    |

> ⚠️ Encrypted files (`*.enc`, `*.key`, `*.pkl`, `*.exe`, `dist/`, `build/`) are excluded from Git via `.gitignore`.


## 🛠️ Build to EXE (Optional)

If you want to create a Windows executable using PyInstaller:

```bash
pyinstaller --onefile --noconsole FaceRecognition.spec
```

   
   
