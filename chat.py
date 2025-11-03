from PyQt5 import QtWidgets, QtGui, QtCore
import threading
import speech_to_text
from datetime import datetime
import os

class ChatWindow(QtWidgets.QWidget):
    user_send_signal = QtCore.pyqtSignal(str)

    def __init__(self, width=828, height=1792, title="AI Chat"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(0, 1440, width, height)
        self.setMinimumSize(256, 400)

        # -- Stacked Layout for "Mic" page and "Chat" page
        self.stack = QtWidgets.QStackedWidget(self)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.stack)
        

        # ---- PAGE 1: Recording page ----
        self.record_page = QtWidgets.QWidget()
        rec_layout = QtWidgets.QVBoxLayout(self.record_page)
        rec_layout.setAlignment(QtCore.Qt.AlignCenter)
        rec_layout.setContentsMargins(32,64,32,64)
        rec_layout.addStretch(3)

        # Large round record button
        self.record_btn = QtWidgets.QPushButton()
        self.record_btn.setFixedSize(240, 240)
        self.record_btn.setCheckable(True)
        self.record_btn.setStyleSheet("""
            QPushButton {
                border-radius: 120px;
                background: #72C529;
                color: white;
                font-size: 48px;
                font-family: 'Garamond', serif;
                font-weight: bold;
            }
        """)
        self.set_record_btn_icon(False)
        self.record_btn.clicked.connect(self.toggle_recording)
        
        rec_layout.addWidget(self.record_btn, alignment=QtCore.Qt.AlignHCenter)

        # "Listening..." label (hidden until recording)
        self.listening_label = QtWidgets.QLabel("")
        self.listening_label.setStyleSheet("color:white; font-family: 'Garamond', serif; font-size: 56px;")
        self.listening_label.setAlignment(QtCore.Qt.AlignHCenter)
        rec_layout.addWidget(self.listening_label)

        rec_layout.addStretch(6)

        # Story button at the bottom
        self.story_btn = QtWidgets.QPushButton("Story")
        self.story_btn.setFixedSize(300, 80)
        self.story_btn.setStyleSheet("""
            QPushButton {
                background: white;
                color: black;
                font-size: 56px;
                font-weight: bold;
                font-family: 'Garamond', serif;
                border-radius: 24px;
            }
        """)
        self.story_btn.clicked.connect(self.show_story_page)
        rec_layout.addWidget(self.story_btn, alignment=QtCore.Qt.AlignHCenter)
        rec_layout.addSpacing(24)

        self.record_page.setStyleSheet("background:#282828;")
        self.stack.addWidget(self.record_page)

        # ---- PAGE 2: Chat/Story page ----
        self.story_page = QtWidgets.QWidget()
        story_layout = QtWidgets.QVBoxLayout(self.story_page)
        story_layout.setContentsMargins(32, 64, 32, 64)
        story_layout.setSpacing(18)

        # Chat history
        self.chat_history = QtWidgets.QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            background: #282828;
            color: white;
            font-size: 40px;
            font-family: 'Garamond', serif;
            border: none;
        """)
        story_layout.addWidget(self.chat_history, stretch=1)

        # Input line (debug/type bar)
        self.input_line = QtWidgets.QLineEdit(self)
        self.input_line.setPlaceholderText("Type your message and press Enter or Send...")
        self.input_line.setMinimumHeight(50)
        self.input_line.setStyleSheet("color:white; font-family: 'Garamond', serif; font-size: 40px;")
        story_layout.addWidget(self.input_line)

        # Record button (returns to record page)
        self.story_record_btn = QtWidgets.QPushButton("Record")
        self.story_record_btn.setFixedSize(300, 80)
        self.story_record_btn.setStyleSheet("""
            QPushButton {
                background: white;
                color: black;
                font-size: 56px;
                font-weight: bold;
                font-family: 'Garamond', serif;
                border-radius: 24px;
            }
        """)
        self.story_record_btn.clicked.connect(self.show_record_page)
        story_layout.addWidget(self.story_record_btn, alignment=QtCore.Qt.AlignHCenter)

        self.story_page.setStyleSheet("background:#282828;")
        self.stack.addWidget(self.story_page)

        # Set default page
        self.show_record_page()

        # --- Signals as before
        self.input_line.returnPressed.connect(self.on_send)
        self._is_recording = False

        # Optional shortcut for record (space)
        shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self)
        shortcut.activated.connect(self.record_btn.click)

    # --- PAGE SWITCHING ---
    def show_story_page(self):
        self.stack.setCurrentWidget(self.story_page)

    def show_record_page(self):
        self.stack.setCurrentWidget(self.record_page)
        # Reset record button and label
        self.record_btn.setChecked(False)
        self.set_record_btn_icon(False)
        self.listening_label.setText("")

    # --- Core chat functions (SAME AS OLD VERSION) ---
    def on_send(self):
        text = self.input_line.text().strip()
        if text:
            self.user_send_signal.emit(text)
            self.append_message("You", text)
            self.input_line.clear()

    def append_message(self, sender, message):
        if sender == "You":
            fmt = '<b style="color:#3266cc">You:</b> '
        else:
            fmt = '<b style="color:#098842">AI:</b> '
        self.chat_history.append(fmt + message)
        self.chat_history.moveCursor(QtGui.QTextCursor.End)

    def append_ai_message(self, message):
        self.append_message("AI", message)

    def append_user_message(self, message):
        self.append_message("You", message)

    def toggle_recording(self):
        if not self._is_recording:
            # Start recording in a thread to avoid blocking the GUI
            self._is_recording = True
            self.set_record_btn_icon(True)
            self.listening_label.setText("Listening...")
            threading.Thread(target=self._start_recording_thread, daemon=True).start()
        else:
            self._is_recording = False
            self.set_record_btn_icon(False)
            self.listening_label.setText("")
            threading.Thread(target=self._stop_and_transcribe_thread, daemon=True).start()

    def _start_recording_thread(self):
        speech_to_text.start_recording()

    def _stop_and_transcribe_thread(self):
        try:
            filename = speech_to_text.stop_recording()
            text = speech_to_text.transcribe_audio(filename)
            QtCore.QMetaObject.invokeMethod(
                self,
                "_send_transcribed_text",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, text)
            )
        except Exception as e:
            def show_error():
                self.input_line.setText(f"[Speech Error: {e}]")
            QtCore.QTimer.singleShot(0, show_error)
    
    def clear_chat(self):
        self.chat_history.clear()  # Or whatever widget displays your chat


    @QtCore.pyqtSlot(str)
    def _send_transcribed_text(self, text):
        # Always auto-switch to story page on result
        #self.show_story_page()
        self.input_line.setText(text)
        self.on_send()

    def set_record_btn_icon(self, listening):
        if listening:
            # Show stop icon
            self.record_btn.setIcon(QtGui.QIcon("ui_elements/stop.png"))
            self.record_btn.setIconSize(QtCore.QSize(120, 120))
            self.record_btn.setText("")  # No text
            self.record_btn.setStyleSheet("""
                QPushButton {
                    border-radius: 120px;
                    background: #e74c3c;
                }
            """)
        else:
            # Show mic icon
            self.record_btn.setIcon(QtGui.QIcon("ui_elements/mic.png"))
            self.record_btn.setIconSize(QtCore.QSize(120, 120))
            self.record_btn.setText("")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    border-radius: 120px;
                    background: #72C529;
                }
            """)
    def save_chat_history(self, folder="chat_history_logs"):
        # Ensure the folder exists
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Find the next file number
        existing = [f for f in os.listdir(folder) if f.startswith("chat_") and f.endswith(".html")]
        if existing:
            nums = [int(f[5:-5]) for f in existing if f[5:-5].isdigit()]
            next_num = max(nums) + 1 if nums else 1
        else:
            next_num = 1

        # Optionally, add a date string for easier human reading
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(folder, f"chat_{next_num}_{date_str}.html")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.chat_history.toHtml())

    def closeEvent(self, event):
        self.save_chat_history()
        super().closeEvent(event)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = ChatWindow()
    win.show()

    def handle_user_msg(msg):
        win.append_ai_message(f"You said: {msg}")

    win.user_send_signal.connect(handle_user_msg)
    sys.exit(app.exec_())
