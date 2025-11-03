import os
import cv2
import asyncio
import json
import aiosqlite
import time
from API import GPT_response
from aruco_id import ARUCO_ID_LABELS
from object_radii_config import OBJECT_RADII
from PyQt5 import QtWidgets, QtCore
from display import StoryMapWindow
import sys
from settings import settings
from chat import ChatWindow
from qasync import QEventLoop, asyncSlot
from tts import say
import threading

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CELL_SIZE = 80

# Detection rectangle parameters
#DETECTION_RECT_X = 0
#DETECTION_RECT_Y = 0
#DETECTION_RECT_W = 1024
#DETECTION_RECT_H = 576


GRID_COLS = 16
GRID_ROWS = 9

CALIBRATION_FILE = "calibration_rect.json"

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeMin = 15
aruco_params.adaptiveThreshWinSizeMax = 35
aruco_params.adaptiveThreshWinSizeStep = 10
aruco_params.adaptiveThreshConstant = 10
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
manager = None

test_ai_reply = """
{
  "message": "The cartoon character is actually an alien, and the blue spot is a lake.",
  "label": {
    "3": "alien",
    "7": "lake"
  },
  "objects": [],
  "remove": []
}


"""

#display config (width=1024, height=576, cell_size=64, x_offset=2688, y_offset=72)

object_labels = {}  # marker_id:int → {'label': str, 'grid': (x, y), 'center': (cx, cy)}

def get_aruco_objects():
    ret, frame = cap.read()
    if not ret:
        return {}, frame

    # Crop to detection rectangle for detection
    x0, y0 = DETECTION_RECT_X, DETECTION_RECT_Y
    x1, y1 = x0 + DETECTION_RECT_W, y0 + DETECTION_RECT_H
    roi = frame[y0:y1, x0:x1]

    corners, ids, _ = cv2.aruco.detectMarkers(roi, aruco_dict, parameters=aruco_params)
    ids_to_obj = {}
    if ids is not None:
        for idx, corner in enumerate(corners):
            marker_id = int(ids[idx][0])
            c = corner[0]
            cx, cy = float(c[:, 0].mean()), float(c[:, 1].mean())
            # Floating-point grid coordinates (e.g., 5.33, 6.89)
            gx = cx / (DETECTION_RECT_W / GRID_COLS)
            gy = cy / (DETECTION_RECT_H / GRID_ROWS)
            abs_corners = c.astype(int) + [x0, y0]
            abs_cx, abs_cy = cx + x0, cy + y0
            ids_to_obj[marker_id] = {
                "center": (abs_cx, abs_cy),
                "grid": (gx, gy),
                "corners": abs_corners
            }
    return ids_to_obj, frame



def draw_grid_on_frame(frame):
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        cv2.line(frame, (x, 0), (x, WINDOW_HEIGHT), (180, 180, 180), 1)
    for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
        cv2.line(frame, (0, y), (WINDOW_WIDTH, y), (180, 180, 180), 1)

def draw_aruco_on_frame(frame, aruco_objs, object_labels):

    for marker_id, obj in aruco_objs.items():
        c = obj["corners"]
        cv2.polylines(frame, [c.astype(int)], True, (0,255,0), 2)  # Always draw the marker boundary
        if marker_id in object_labels:
            cx, cy = obj["center"]
            grid_x, grid_y = obj["grid"]
            label = object_labels[marker_id]
            cv2.putText(frame, f"{label} ({marker_id}) [{grid_x:.2f},{grid_y:.2f}]",
                        (int(cx)+16, int(cy)-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,180,0), 2)



def describe_location(grid_x, grid_y, cols=16, rows=9):
    horiz = "left" if grid_x < cols / 3 else "middle" if grid_x < 2 * cols / 3 else "right"
    vert = "top" if grid_y < rows / 3 else "middle" if grid_y < 2 * rows / 3 else "bottom"
    return f"{vert} {horiz}"

def draw_detection_rectangle(frame):
    x0, y0 = DETECTION_RECT_X, DETECTION_RECT_Y
    x1, y1 = x0 + DETECTION_RECT_W, y0 + DETECTION_RECT_H
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

def save_calibration_rect(x, y, w, h):
    with open(CALIBRATION_FILE, "w") as f:
        json.dump({"x": x, "y": y, "w": w, "h": h}, f)

def load_calibration_rect():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)
            return data["x"], data["y"], data["w"], data["h"]
    # Return your defaults if file not found
    return 0, 0, 1024, 576

DETECTION_RECT_X, DETECTION_RECT_Y, DETECTION_RECT_W, DETECTION_RECT_H = load_calibration_rect()
print(f"Detection rectangle: {DETECTION_RECT_X}, {DETECTION_RECT_Y}, {DETECTION_RECT_W}, {DETECTION_RECT_H}")


class AppManager(QtCore.QObject):
    def __init__(self, chat_win, grid_win, ai_callback, chat_id = None, story_thread_id = None, supervisor_thread_id = None, user_id = None):
        super().__init__()
        self.chat_win = chat_win
        self.grid_win = grid_win
        self.ai_callback = ai_callback
        self.chat_id = chat_id
        self.story_thread_id = story_thread_id
        self.supervisor_thread_id = supervisor_thread_id
        self.user_id = user_id

        # Timer to update grid/camera
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_grid_display)
        self.timer.start(100)  # update every 100 ms

        self.supervisor_advice = None
        self._supervisor_task = None

        self.chat_history = []  # List of (role, text), e.g. ("user", "..."), ("ai", "...")

        self.history_window = []  # List of (user_message, story_ai_reply) tuples
        self.history_window_size = 2  # Adjustable

        self.virtual_objects = []  # Each is a dict: {"cell": [x, y], "label": "...", "color": "..."}

        self.object_labels = {}  # marker_id : label

        self.prev_physical_ids = set()  # IDs of objects seen in the previous round

        self.detection_buffer = []   # List of (timestamp, detected_objects_dict)
        self.detection_start_time = None
        self.DETECTION_PERIOD = 1.0  # seconds

        # Chat signal
        self.chat_win.user_send_signal.connect(self.process_user_message)

        self.marker_presence = {}  # marker_id -> consecutive frames detected
        self.marker_absence = {}   # marker_id -> consecutive frames NOT detected
        self.PRESENCE_THRESHOLD = 5  # Number of frames before considering "present"
        self.ABSENCE_THRESHOLD = 10   # Number of missed frames before considered "gone"
        self.last_seen_objects = {}



    async def start_session(self):
        import aiosqlite
        import time
        db_path = "tianrui_demo.db"
        db_connection = await aiosqlite.connect(db_path)
        self.chat_id = int(time.time())
        from API import create_new_thread
        self.story_thread_id, self.user_id = await create_new_thread(self.chat_id, db_connection)
        self.supervisor_thread_id, _ = await create_new_thread(self.chat_id + 1, db_connection)
        await db_connection.close()

        self.chat_history = []

        # Clear all session state on restart
        self.object_labels = {}
        self.virtual_objects = []
        self.chat_history = []

        # Also clear the display and chat window:
        self.grid_win.update_objects({}, labels={})  # Remove all labeled objects
        self.grid_win.update_virtual_objects([])     # Remove all virtual objects

        if hasattr(self.chat_win, "/restart"):
            self.chat_win.clear_chat()


    def update_grid_display(self):
        global DETECTION_RECT_X, DETECTION_RECT_Y, DETECTION_RECT_W, DETECTION_RECT_H
        aruco_objs, frame = get_aruco_objects()
        object_positions = {marker_id: obj['grid'] for marker_id, obj in aruco_objs.items()}
        #self.grid_win.widget.set_object_radii(OBJECT_RADII)
        
        current_ids = set(aruco_objs.keys())
        # Update presence/absence counters
        for marker_id in current_ids:
            self.marker_presence[marker_id] = self.marker_presence.get(marker_id, 0) + 1
            self.marker_absence[marker_id] = 0

        for marker_id in list(self.marker_presence):
            if marker_id not in current_ids:
                self.marker_absence[marker_id] = self.marker_absence.get(marker_id, 0) + 1
                if self.marker_absence[marker_id] > self.ABSENCE_THRESHOLD:
                    # Remove if absent for too long
                    self.marker_presence.pop(marker_id)
                    self.marker_absence.pop(marker_id)

        # Build filtered object list (only those seen for enough frames, and not absent for too long)
        filtered_object_positions = {}
        for marker_id, count in self.marker_presence.items():
            if count >= self.PRESENCE_THRESHOLD and self.marker_absence.get(marker_id, 0) <= self.ABSENCE_THRESHOLD:
                filtered_object_positions[marker_id] = aruco_objs.get(marker_id, None) or self.last_seen_objects.get(marker_id, None)

        # Optionally, store last seen positions for smooth display
        self.last_seen_objects = {**self.last_seen_objects, **aruco_objs}
        
        visible_objects = {}
        for marker_id, grid in object_positions.items():
            if marker_id in self.object_labels:
                visible_objects[marker_id] = grid

        self.grid_win.update_objects(
            visible_objects,  # Only labeled physical objects
            labels={mid: self.object_labels[mid] for mid in visible_objects}
        )
        draw_aruco_on_frame(frame, aruco_objs, self.object_labels)
        draw_detection_rectangle(frame)
        cv2.imshow("Debug Camera", frame)
        key = cv2.waitKey(1)
        ZOOM_STEP = 20
        if key == ord('a'):  # left
            DETECTION_RECT_X = max(0, DETECTION_RECT_X - 5)
        if key == ord('d'):  # right
            DETECTION_RECT_X += 5
        if key == ord('w'):  # up
            DETECTION_RECT_Y = max(0, DETECTION_RECT_Y - 5)
        if key == ord('s'):  # down
            DETECTION_RECT_Y += 5
        if key == ord('q'):  # zoom in (shrink, keep ratio)
            DETECTION_RECT_W = max(32, DETECTION_RECT_W - ZOOM_STEP)
            DETECTION_RECT_H = int(DETECTION_RECT_W * GRID_ROWS / GRID_COLS)
        if key == ord('e'):  # zoom out (expand, keep ratio)
            DETECTION_RECT_W += ZOOM_STEP
            DETECTION_RECT_H = int(DETECTION_RECT_W * GRID_ROWS / GRID_COLS)
        if key == ord('c'):  # save calibration
            save_calibration_rect(DETECTION_RECT_X, DETECTION_RECT_Y, DETECTION_RECT_W, DETECTION_RECT_H)
            print(f"Saved calibration rectangle: {DETECTION_RECT_X}, {DETECTION_RECT_Y}, {DETECTION_RECT_W}, {DETECTION_RECT_H}")
        
        if self.detection_start_time is not None and not getattr(self, "detection_finalized", False):
            # Buffer each detection snapshot
            aruco_objs_snapshot = {marker_id: obj.copy() for marker_id, obj in aruco_objs.items()}
            self.detection_buffer.append((time.time(), aruco_objs_snapshot))

            if (time.time() - self.detection_start_time) >= self.DETECTION_PERIOD:
                # Detection period over: select best snapshot
                if self.detection_buffer:
                    best_snapshot = max(self.detection_buffer, key=lambda tup: len(tup[1]))
                    self.confirmed_world_state = best_snapshot[1]
                    print(f"[Detection] Finalized with {len(self.confirmed_world_state)} objects.")
                else:
                    self.confirmed_world_state = {}
                    print("[Detection] No objects detected.")
                self.detection_finalized = True
                self.detection_start_time = None


    @asyncSlot(str)
    async def process_user_message(self, user_message):
        if user_message.strip().lower() == "/restart":
            await self.start_session()
            #self.chat_win.append_user_message("/restart")
            return

        self.detection_buffer = []
        self.detection_start_time = time.time()
        self.detection_finalized = False
        # Wait for previous supervisor task (if running) before composing prompt
        if self._supervisor_task is not None and not self._supervisor_task.done():
            await self._supervisor_task

        db_path = "tianrui_demo.db"
        db_connection = await aiosqlite.connect(db_path)
        chat_id = self.chat_id

        # Wait until detection is finalized
        while not getattr(self, "detection_finalized", False):
            await asyncio.sleep(0.01)

        aruco_objs = getattr(self, "confirmed_world_state", {})

        obj_msgs = []

        current_physical_ids = set(aruco_objs.keys())

        for marker_id, obj in aruco_objs.items():
            # Check if this object is new this round
            is_new = marker_id not in self.prev_physical_ids
            # Use label if assigned, otherwise default name
            label = self.object_labels.get(marker_id) or ARUCO_ID_LABELS.get(marker_id, f"object {marker_id}")
            gx, gy = obj['grid']
            rel_loc = describe_location(gx, gy)
            new_str = "(new) " if is_new else ""
            obj_msgs.append(
                f"{new_str}{label} (ID: {marker_id}) at ({gx:.2f}, {gy:.2f}), on the {rel_loc} of the map"
            )

        # After building, update for the next round:
        self.prev_physical_ids = current_physical_ids


        # Virtual objects
        for vobj in self.virtual_objects:
            vlabel = vobj.get("label", "virtual object")
            vx, vy = vobj.get("cell", [None, None])
            vloc = describe_location(vx, vy)
            obj_msgs.append(f"{vlabel} (virtual) at ({vx}, {vy}), on the {vloc} of the map")

        scene_desc = (
            "Current world state:\n- " +
            "\n- ".join(obj_msgs) +
            f"\nUser message: {user_message}\n"
        )
        prompt = ""
        if getattr(self, "supervisor_advice", ""):
            prompt += f"Supervisor system tip: {self.supervisor_advice}\n"
        prompt += scene_desc
        
        print(f"Sent to AI:{prompt}")

        ai_reply = await GPT_response(
            prompt, self.chat_id, db_connection,
            assistant_id=settings["STORY_AI_ID"],
            thread_id=self.story_thread_id
        )
        print(f"AI:{ai_reply}")
        # After getting ai_reply from the story AI
        self.chat_history.append(("user", user_message))

        #self.history_window.append((user_message, ai_reply))
        # Keep only the last N rounds
        #if len(self.history_window) > self.history_window_size:
        #    self.history_window = self.history_window[-self.history_window_size:]

        try:
            resp = json.loads(ai_reply)
            # Handle adding virtual objects
            if "virtual objects" in resp:
                # Add new or update existing
                for obj in resp["virtual objects"]:
                    # Check if already exists by cell or label; if not, add it
                    already = any((obj["cell"] == vobj["cell"] and obj["label"] == vobj["label"]) for vobj in self.virtual_objects)
                    if not already:
                        self.virtual_objects.append(obj)

            label_dict = resp.get("label", {})
            for marker_id_str, label in label_dict.items():
                if marker_id_str.isdigit() and label:
                    marker_id = int(marker_id_str)
                    self.object_labels[marker_id] = label




            # Handle removal if specified (e.g., AI sends "remove": [{"cell": [x, y], ...}] )
            if "remove" in resp:
                for to_remove in resp["remove"]:
                    self.virtual_objects = [
                        vobj for vobj in self.virtual_objects
                        if not (vobj["cell"] == to_remove["cell"] and vobj["label"] == to_remove["label"])
                    ]

            # (Draw all objects: physical and self.virtual_objects as needed)
            self.grid_win.update_virtual_objects(self.virtual_objects)
            self.chat_win.append_ai_message(resp.get("message", "[No message from AI]"))
            ai_text = resp.get("message", "[No message from AI]")
            print(f"AI:{ai_text}")
            self.chat_history.append(("ai", ai_text))

            if ai_text:
                threading.Thread(target=say, args=(ai_text,), daemon=True).start()

        except Exception as e:
            self.chat_win.append_ai_message(f"[AI response error: {e}]")


        # Start supervisor analysis for this new AI reply (as background task)
        self._supervisor_task = asyncio.create_task(self.get_supervisor_advice(ai_reply, chat_id))
        await db_connection.close()
        self.detection_finalized = False
        self.confirmed_world_state = {}


    async def get_supervisor_advice(self, story_reply, chat_id, db_connection=None):
        import aiosqlite
        # Always open a new connection for this background task
        db_connection = await aiosqlite.connect("tianrui_demo.db")
        try:
            # Build prompt with last N rounds (user + story AI)
            #prompt_lines = []
            #for user_msg, story_ai_msg in self.history_window:
            #    prompt_lines.append(f"User: {user_msg}")
            #    prompt_lines.append(f"Story AI: {story_ai_msg}")

            dialogue = ""
            for role, text in self.chat_history:
                if role == "user":
                    dialogue += f"User: {text}\n"
                elif role == "ai":
                    dialogue += f"Story AI: {text}\n"


            supervisor_prompt = (
                "Here is the full story conversation so far:\n"
                f"{dialogue}\n"
                "Please analyze and give your advice for the next move."
            )

            supervisor_reply = await GPT_response(
                supervisor_prompt, self.chat_id, db_connection,
                assistant_id=settings["SUPERVISOR_AI_ID"],
                thread_id=self.supervisor_thread_id
            )

            print(f"Supervisor AI:{supervisor_reply}")
            try:
                reply_json = json.loads(supervisor_reply)
                self.supervisor_advice = reply_json.get("message", "")
            except Exception:
                self.supervisor_advice = supervisor_reply
        finally:
            await db_connection.close()

    def closeEvent(self, event):
        global DETECTION_RECT_X, DETECTION_RECT_Y, DETECTION_RECT_W, DETECTION_RECT_H
        save_calibration_rect(DETECTION_RECT_X, DETECTION_RECT_Y, DETECTION_RECT_W, DETECTION_RECT_H)
        print(f"Saved calibration rectangle: {DETECTION_RECT_X}, {DETECTION_RECT_Y}, {DETECTION_RECT_W}, {DETECTION_RECT_H}")
        event.accept()





# ---- Main entrypoint ----
if __name__ == "__main__":
    from qasync import QEventLoop
    import aiosqlite
    import time

    app = QtWidgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    grid_win = StoryMapWindow(width=1152, height=648, cell_size=72, x_offset=2624, y_offset=36)
    chat_win = ChatWindow(width=828, height=1792, title="AI Chat")
    grid_win.show()
    chat_win.show()
    
    async def init_new_session():
        db_connection = await aiosqlite.connect("tianrui_demo.db")
        chat_id = int(time.time())
        from API import create_new_thread
        story_thread_id, user_id = await create_new_thread(chat_id, db_connection)
        supervisor_thread_id, _ = await create_new_thread(chat_id + 1, db_connection)
        await db_connection.close()
        return chat_id, story_thread_id, supervisor_thread_id, user_id

    async def startup():
        global manager
        manager = AppManager(chat_win, grid_win, GPT_response, None, None, None, None)
        await manager.start_session()



    with loop:
        loop.run_until_complete(startup())
        # Wait for user input to start the first message
        input("Press Enter to begin the conversation and send 'hi' to the assistant...")
        loop.run_until_complete(manager.process_user_message("hi"))
        loop.run_forever()


# To-Do: 
# 1. add virtual objects to world description, and don't delete the virtual objects every round.
# 2. change display, bigger circle, more virtual display modes?
# 3. virtual display logic: a river: long, multiple cells.
# 4. system robustness: multiple times detection.
