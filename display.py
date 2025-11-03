from PyQt5 import QtWidgets, QtGui, QtCore
import sys
from object_radii_config import OBJECT_RADII


class StoryMapWidget(QtWidgets.QWidget):
    def __init__(self, width=1280, height=800, cell_size=80, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setWindowTitle("Story Map - User Display")
        self.resize(width, height)
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid_cols = width // cell_size
        self.grid_rows = height // cell_size
        self.object_positions = {}  # {marker_id: (grid_x, grid_y)}
        self.labels = {}  # {marker_id: label string}
        self.object_radii = {}

    def update_objects(self, object_positions, labels=None):
        self.object_positions = object_positions or {}
        if labels:
            self.labels = labels
        self.update()  # Trigger repaint

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        # background
        qp.fillRect(0, 0, self.width, self.height, QtGui.QColor('black'))
        qp.setRenderHint(QtGui.QPainter.Antialiasing)

        # Only draw labels for objects present in self.labels (provided by manager)
        for marker_id, (gx, gy) in self.object_positions.items():
            if marker_id in self.labels and self.labels[marker_id]:
                label = self.labels[marker_id]
                cx = gx * self.cell_size + self.cell_size // 2
                cy = gy * self.cell_size + self.cell_size // 2
                # --- Offset label downwards by the object's radius (default to e.g. 30 if missing) ---
                radius = OBJECT_RADII.get(marker_id, 30)
                # Draw label under the position
                qp.setPen(QtGui.QColor('#fff78c'))
                font = QtGui.QFont("Garamond", 18, QtGui.QFont.Bold)
                qp.setFont(font)
                font_metrics = QtGui.QFontMetrics(font)
                label_width = font_metrics.horizontalAdvance(label)
                label_height = font_metrics.height()
                text_x = cx - label_width // 2
                text_y = cy + radius/2  # slightly below center
                qp.drawText(QtCore.QRectF(text_x, text_y, label_width, label_height),
                            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, label)
     
        # Draw virtual objects
        for obj in getattr(self, "virtual_objects", []):
            x, y = obj["cell"]
            color = QtGui.QColor(obj.get("color", "#99d0ff"))
            label = obj.get("label", "")
            cx = x * self.cell_size + self.cell_size // 2
            cy = y * self.cell_size + self.cell_size // 2
            radius = int(self.cell_size * 0.35)
            self.draw_glowing_circle(qp, cx, cy, radius, color, glow_width=18)
            if label:
                qp.setPen(QtGui.QColor('#d0eaff'))
                font = QtGui.QFont("Garamond", 18, QtGui.QFont.Bold)
                qp.setFont(font)

                # Use font metrics to measure label width
                font_metrics = QtGui.QFontMetrics(font)
                label_width = font_metrics.horizontalAdvance(label)
                label_height = font_metrics.height()

                # Center label horizontally under the circle
                text_x = cx - label_width // 2
                text_y = cy + radius + 5  # 5 px below the circle

                # Draw the label in a rectangle wide enough for the label
                qp.drawText(QtCore.QRectF(text_x, text_y, label_width, label_height),
                            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, label)


    def update_virtual_objects(self, objects):
        """
        objects: list of dicts, each with 'cell': [x, y], 'label', 'color'
        """
        self.virtual_objects = objects
        self.update()

    def draw_glowing_circle(self, qp, cx, cy, radius, color, glow_width=0):
        color.setAlpha(255)
        qp.setBrush(color)
        qp.setPen(QtCore.Qt.NoPen)
        qp.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

    def set_object_radii(self, radii_dict):
        self.object_radii = radii_dict
        self.update()



class StoryMapWindow(QtWidgets.QMainWindow):
    def __init__(self, width=1280, height=800, cell_size=80, x_offset=0, y_offset=0):
        super().__init__()
        self.setGeometry(x_offset, y_offset, width, height)
        self.setWindowTitle("Story Map - User Display (PyQt)")
        self.widget = StoryMapWidget(width, height, cell_size)
        self.setCentralWidget(self.widget)

    def update_objects(self, object_positions, labels=None):
        self.widget.update_objects(object_positions, labels)
    
    def update_virtual_objects(self, objects):
        self.widget.update_virtual_objects(objects)

# --- For testing standalone ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = StoryMapWindow()
    win.show()

    import random

    def anim():
        # Randomly place 3 markers every 200ms
        obj_pos = {i: (random.randint(0,15), random.randint(0,9)) for i in range(3)}
        win.update_objects(obj_pos)
        # QTimer will call this again

    timer = QtCore.QTimer()
    timer.timeout.connect(anim)
    timer.start(200)  # every 200 milliseconds

    sys.exit(app.exec_())

