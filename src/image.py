import cv2 as cv
import numpy as np
import math

# Color mapping
COLOR_MAP = {
    0: (47, 37, 30),      # background
    1: (128, 128, 128),   # gray
    2: (113, 214, 59),    # green
    3: (92, 92, 255),     # red
    4: (102, 209, 255),   # yellow
    5: (209, 158, 43),    # blue
    6: (36, 27, 21),      # super-dark blue
}

# Image and box parameters
IMG_PADDING = 30
IMG_MARGIN = 5
BOX_SIZE = 50
BOX_RADIUS = 3

class ImageProcessor:
    """Handles image to matrix conversion and visualization"""
    
    def __init__(self):
        self.last_img_bgr = None
        self.last_box_centers = {}
        self.original_img_bgr = None

    @staticmethod
    def draw_rounded_box(image, pt1, pt2, color, radius=BOX_RADIUS):
        """Draw a filled rounded rectangle on the image"""
        x1, y1 = pt1
        x2, y2 = pt2
        overlay = np.zeros_like(image, dtype=np.uint8)

        cv.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        
        mask = overlay.astype(bool)
        image[mask] = overlay[mask]

    def draw_star(self, img, center, outer_radius, inner_radius, color):
        """Draws a 5-pointed star centered at `center`."""
        cx, cy = center
        pts = []
        for i in range(10):
            angle = i * math.pi / 5 - math.pi / 2
            r = outer_radius if i % 2 == 0 else inner_radius
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            pts.append((x, y))
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv.fillPoly(img, [pts], color)

    def _read_image_flex(self, source):
        """Accepts either a numpy array (BGR), raw bytes, or a filesystem path and
        returns a BGR numpy array.
        """
        if isinstance(source, np.ndarray):
            return source
        if isinstance(source, (bytes, bytearray)):
            arr = np.frombuffer(source, dtype=np.uint8)
            img = cv.imdecode(arr, cv.IMREAD_COLOR)
            return img
        if isinstance(source, str):
            return cv.imread(source)
        raise TypeError("Unsupported image source type")
    
    def img_to_matrix(self, image_path):
        """Convert image to a matrix representation"""
        img = self._read_image_flex(image_path)
        self.last_img_bgr = img.copy()

        # Preprocessing
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        _, thresh = cv.threshold(v_channel, 80, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Filter bbox
        boxes = [cv.boundingRect(c) for c in contours if cv.contourArea(c) > 100]
        boxes = [b for b in boxes if b[2] > 20 and b[3] > 20]
        boxes.sort(key=lambda b: (b[1] // 50, b[0]))

        # Index map
        ys = sorted(set(b[1] // 50 for b in boxes))
        xs = sorted(set(b[0] // 50 for b in boxes))
        matrix = np.zeros((len(ys), len(xs)), dtype=int)

        # Matrix generation
        for (x, y, w, h) in boxes:
            roi_hsv = hsv[y:y+h, x:x+w]
            mean_h, mean_s, mean_v = np.mean(roi_hsv.reshape(-1, 3), axis=0)
            val = 2 if mean_s > 50 and mean_v > 60 else 1
            row, col = ys.index(y // 50), xs.index(x // 50)
            matrix[row, col] = val
            self.last_box_centers[(row, col)] = (x + w / 2, y + h / 2)

        return matrix
    
    def generate_img(self, matrix):
        rows, cols = matrix.shape
        img_h = rows * (BOX_SIZE + IMG_MARGIN) + IMG_MARGIN
        img_w = cols * (BOX_SIZE + IMG_MARGIN) + IMG_MARGIN

        # Background initialization
        bg_img = np.full((img_h, img_w, 3), COLOR_MAP[0], dtype=np.uint8)

        # Draw boxes
        for r in range(rows):
            for c in range(cols):
                val = matrix[r, c]
                color = COLOR_MAP.get(val, (128, 128, 128))
                x1 = c * (BOX_SIZE + IMG_MARGIN) + IMG_MARGIN
                y1 = r * (BOX_SIZE + IMG_MARGIN) + IMG_MARGIN
                x2, y2 = x1 + BOX_SIZE, y1 + BOX_SIZE
                self.draw_rounded_box(bg_img, (x1, y1), (x2, y2), color, radius=BOX_RADIUS)

                cx = x1 + BOX_SIZE / 2 + IMG_PADDING
                cy = y1 + BOX_SIZE / 2 + IMG_PADDING
                self.last_box_centers[(r, c)] = (cx, cy)

        # Add padding
        img = cv.copyMakeBorder(bg_img, IMG_PADDING, IMG_PADDING, IMG_PADDING, IMG_PADDING,
                                        cv.BORDER_CONSTANT, value=COLOR_MAP[0])
        self.last_img_bgr = img.copy()

        return img

    def draw_path_on_image(self, matrix, path, start, finish):
        """Draw the solution path on the last loaded image"""
        if self.last_img_bgr is None:
            raise ValueError("No image loaded. Call img_to_matrix first.")
        
        self.original_img_bgr = self.last_img_bgr.copy()
        new_matrix = matrix.copy()
        for i in range(1, len(path)-1):
            r, c = path[i]
            new_matrix[r, c] = 4
        finish_r, finish_c = finish
        new_matrix[finish_r, finish_c] = 3
        img = self.generate_img(new_matrix)

        # Prepare points
        pts = []
        for node in path:
            if node in self.last_box_centers:
                pts.append(tuple(map(int, self.last_box_centers[node])))
            else:
                pts.append(None)

        # Draw connecting lines
        for i in range(1, len(pts)):
            p0 = pts[i-1]
            p1 = pts[i]
            if p0 is None or p1 is None:
                continue
            cv.line(img, p0, p1, color=(255, 0, 0), thickness=6, lineType=cv.LINE_AA)

        # Draw node markers
        for i, node in enumerate(path):
            if node not in self.last_box_centers:
                continue
            cx, cy = map(int, self.last_box_centers[node])
            if i == 0 or node == start:
                # Start node
                cv.circle(img, (cx, cy), radius=16, color=COLOR_MAP[5], 
                            thickness=-1, lineType=cv.LINE_AA)
                cv.circle(img, (cx, cy), radius=9, color=COLOR_MAP[6], 
                            thickness=-1, lineType=cv.LINE_AA)
            elif node == finish:
                # Finish node
                cv.circle(img, (cx, cy), radius=16, color=COLOR_MAP[5], 
                            thickness=-1, lineType=cv.LINE_AA)
                self.draw_star(img, (cx, cy), outer_radius=10, inner_radius=5, color=COLOR_MAP[6])

        return img