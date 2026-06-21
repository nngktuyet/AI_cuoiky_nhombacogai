import os
import uuid
import base64
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from flask import Flask, request, jsonify, render_template_string, send_from_directory

# =========================
# OPTIONAL MODEL IMPORTS
# =========================
# App vẫn mở được giao diện nếu thiếu thư viện/model, nhưng khi đủ file thì sẽ tự load.
try:
    import cv2
    import numpy as np
except Exception as e:
    cv2 = None
    np = None
    print("[WARN] Chưa import được cv2/numpy:", e)

try:
    import tensorflow as tf
except Exception as e:
    tf = None
    print("[WARN] Chưa import được tensorflow:", e)


# =========================
# APP CONFIG
# =========================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ORIGINAL_DIR = os.path.join(UPLOAD_DIR, "originals")
CROP_DIR = os.path.join(UPLOAD_DIR, "crops")
DEBUG_DIR = os.path.join(UPLOAD_DIR, "debug")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ASSET_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)

# Model mới theo logic đã chốt với người dùng:
# - model_food.h5: CNN chính nhận diện 14 món.
# - model_thitkho1.h5: CNN phụ kiểm tra ô crop trong thịt kho có trứng hay không.
CNN1_MODEL_PATH = os.path.join(MODEL_DIR, "model_food.h5")
CNN1_CLASS_PATH = os.path.join(MODEL_DIR, "class_indices.json")
EGG_CNN_MODEL_PATH = os.path.join(MODEL_DIR, "model_thitkho1.h5")
EGG_CLASS_PATH = os.path.join(MODEL_DIR, "egg_class_indices.json")

IMG_SIZE_CNN = (128, 128)
CNN_CONFIDENCE_THRESHOLD = 0.60  # Chỉ lưu để tham khảo, giao diện không đánh dấu cảnh báo.
EGG_CONFIDENCE_THRESHOLD = 0.80
EGG_GRID_ROWS = 5
EGG_GRID_COLS = 6
MAX_EGGS = 5
APPLY_GLARE_PREPROCESSING = True

# =========================
# MENU PRICE
# Key đã normalize bằng safe_label(): chữ thường, thay "-" và space bằng "_".
# =========================
MENU_PRICE = {
    "com": 10000,
    "com_trang": 10000,

    "dau_sot_ca": 25000,
    "dau_hu_sot_ca": 25000,
    "ca_hu_kho": 30000,
    "ca_hu_kho_to": 30000,
    "thit_kho": 25000,
    "thit_kho_trung": 30000,
    "suon": 30000,
    "suon_nuong": 30000,

    "canh_chua_co_ca": 25000,
    "canh_chua_khong_ca": 10000,
    "canh_rau": 7000,
    "canh_rau_cai": 7000,
    "canh_rau_muong": 7000,

    "rau_xao": 10000,
    "lagim_xao": 10000,
    "cu_san_xao": 10000,
    "dau_dua_xao": 10000,
    "dau_que_xao": 10000,

    "trung_chien": 25000,
    "trung_chien_thit": 25000,
}

DISPLAY_NAME = {
    "com": "Cơm trắng",
    "com_trang": "Cơm trắng",
    "dau_sot_ca": "Đậu hũ sốt cà",
    "dau_hu_sot_ca": "Đậu hũ sốt cà",
    "ca_hu_kho": "Cá hú kho",
    "ca_hu_kho_to": "Cá hú kho",
    "thit_kho": "Thịt kho",
    "thit_kho_trung": "Thịt kho trứng",
    "canh_chua_co_ca": "Canh chua có cá",
    "canh_chua_khong_ca": "Canh chua không cá",
    "suon": "Sườn nướng",
    "suon_nuong": "Sườn nướng",
    "canh_rau": "Canh rau",
    "canh_rau_cai": "Canh rau cải",
    "canh_rau_muong": "Canh rau muống",
    "rau_xao": "Rau xào",
    "lagim_xao": "Lagim xào",
    "cu_san_xao": "Củ sắn xào",
    "dau_dua_xao": "Đậu đũa xào",
    "dau_que_xao": "Đậu que xào",
    "trung_chien": "Trứng chiên",
    "trung_chien_thit": "Trứng chiên thịt",
}

# =========================
# DEFAULT ROI BOXES
# Các box mặc định theo khay mẫu 1920 x 1080.
# Frontend hiển thị 5 box này dưới dạng tỷ lệ %, người dùng có thể kéo/sửa/xóa.
# =========================
TEMPLATE_SIZE = (1920, 1080)  # width, height
TRAY_ROIS: Dict[str, Tuple[int, int, int, int]] = {
    "top_left":      (340, 100, 925, 535),
    "top_right":     (1080, 100, 1595, 535),
    "bottom_left":   (345, 585, 700, 900),
    "bottom_center": (760, 605, 1125, 920),
    "bottom_right":  (1160, 585, 1535, 900),
}

ROI_DISPLAY_NAME = {
    "top_left": "Box 1",
    "top_right": "Box 2",
    "bottom_left": "Box 3",
    "bottom_center": "Box 4",
    "bottom_right": "Box 5",
}


# =========================
# LOAD MODELS ONCE
# =========================
cnn1_model = None
cnn1_classes: List[str] = []
egg_cnn_model = None
egg_cnn_classes: List[str] = []


def safe_label(label: str) -> str:
    """Chuẩn hóa tên class: chữ thường, bỏ khoảng trắng, đổi - thành _."""
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def load_class_list(class_path: str) -> List[str]:
    """
    Hỗ trợ 3 dạng class file:
    1) {"com": 0, "canh_chua": 1}
    2) {"0": "com", "1": "canh_chua"}
    3) ["com", "canh_chua"]
    """
    if not os.path.exists(class_path):
        print("[WARN] Không tìm thấy class file:", class_path)
        return []

    with open(class_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [safe_label(str(x)) for x in data]

    if not isinstance(data, dict):
        raise ValueError("Class file phải là dict hoặc list")

    if all(str(k).isdigit() for k in data.keys()):
        classes = []
        for i in range(len(data)):
            classes.append(safe_label(str(data[str(i)])))
        return classes

    if all(isinstance(v, int) for v in data.values()):
        classes = [None] * len(data)
        for class_name, idx in data.items():
            classes[int(idx)] = safe_label(str(class_name))
        return classes

    raise ValueError("Class file không đúng format. Cần dạng {'0':'class'} hoặc {'class':0}")


def load_models():
    global cnn1_model, cnn1_classes, egg_cnn_model, egg_cnn_classes

    if tf is not None and os.path.exists(CNN1_MODEL_PATH):
        try:
            cnn1_model = tf.keras.models.load_model(CNN1_MODEL_PATH, compile=False)
            print("[OK] Loaded CNN món ăn:", CNN1_MODEL_PATH)
        except Exception as e:
            print("[ERROR] Không load được CNN món ăn:", e)
    else:
        print("[WARN] Chưa thấy CNN món ăn:", CNN1_MODEL_PATH)

    try:
        cnn1_classes = load_class_list(CNN1_CLASS_PATH)
        print("[OK] CNN món ăn classes:", cnn1_classes)
    except Exception as e:
        print("[ERROR] Không load được class_indices.json:", e)
        cnn1_classes = []

    if tf is not None and os.path.exists(EGG_CNN_MODEL_PATH):
        try:
            egg_cnn_model = tf.keras.models.load_model(EGG_CNN_MODEL_PATH, compile=False)
            print("[OK] Loaded CNN kiểm tra trứng:", EGG_CNN_MODEL_PATH)
        except Exception as e:
            print("[ERROR] Không load được CNN kiểm tra trứng:", e)
    else:
        print("[WARN] Chưa thấy CNN kiểm tra trứng:", EGG_CNN_MODEL_PATH)

    try:
        egg_cnn_classes = load_class_list(EGG_CLASS_PATH)
        print("[OK] CNN kiểm tra trứng classes:", egg_cnn_classes)
    except Exception as e:
        print("[ERROR] Không load được egg_class_indices.json:", e)
        egg_cnn_classes = []


load_models()


# =========================
# UTILS
# =========================
def vnd(amount: int) -> str:
    return f"{amount:,}".replace(",", ".") + " đ"


def save_upload_file(file_storage) -> str:
    ext = os.path.splitext(file_storage.filename or "")[1].lower() or ".jpg"
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(ORIGINAL_DIR, filename)
    file_storage.save(path)
    return path


def save_base64_image(data_url: str) -> str:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(ORIGINAL_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


def normalize_token(token: str) -> str:
    return os.path.basename(str(token or ""))


def get_original_path_from_token(token: str) -> str:
    safe_token = normalize_token(token)
    if not safe_token:
        raise ValueError("Thiếu image_token")
    path = os.path.join(ORIGINAL_DIR, safe_token)
    if not os.path.exists(path):
        raise FileNotFoundError("Không tìm thấy ảnh đã lưu. Hãy tải/chụp lại ảnh.")
    return path


def default_roi_boxes() -> List[Dict]:
    template_w, template_h = TEMPLATE_SIZE
    boxes = []
    for idx, (slot_name, (x1, y1, x2, y2)) in enumerate(TRAY_ROIS.items(), start=1):
        boxes.append({
            "id": slot_name,
            "slot": slot_name,
            "title": ROI_DISPLAY_NAME.get(slot_name, f"Box {idx}"),
            "x": round(x1 / template_w, 6),
            "y": round(y1 / template_h, 6),
            "w": round((x2 - x1) / template_w, 6),
            "h": round((y2 - y1) / template_h, 6),
        })
    return boxes


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sanitize_boxes(raw_boxes: List[Dict]) -> List[Dict]:
    if not isinstance(raw_boxes, list) or not raw_boxes:
        raise ValueError("Bạn cần giữ lại ít nhất 1 box để nhận diện")

    boxes = []
    for i, box in enumerate(raw_boxes, start=1):
        if not isinstance(box, dict):
            continue
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        w = float(box.get("w", 0))
        h = float(box.get("h", 0))
        x = clamp(x, 0.0, 0.98)
        y = clamp(y, 0.0, 0.98)
        w = clamp(w, 0.03, 1.0 - x)
        h = clamp(h, 0.03, 1.0 - y)
        slot = safe_label(box.get("slot") or box.get("id") or f"box_{i}")
        title = str(box.get("title") or f"Box {i}")
        boxes.append({"slot": slot, "title": title, "x": x, "y": y, "w": w, "h": h})

    if not boxes:
        raise ValueError("Không có box hợp lệ để nhận diện")
    return boxes


def crop_custom_boxes(image_path: str, request_id: str, boxes: List[Dict]) -> List[Dict]:
    """
    Crop theo box người dùng đã chỉnh trên giao diện.
    Box dùng tọa độ tỷ lệ 0..1 nên phù hợp với ảnh upload/camera nhiều kích thước khác nhau.
    """
    if cv2 is None:
        raise RuntimeError("Bạn cần cài opencv-python: pip install opencv-python")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    img_h, img_w = image.shape[:2]
    boxes = sanitize_boxes(boxes)

    this_crop_dir = os.path.join(CROP_DIR, request_id)
    os.makedirs(this_crop_dir, exist_ok=True)

    crops = []
    debug = image.copy()

    for index, box in enumerate(boxes, start=1):
        slot = safe_label(box["slot"] or f"box_{index}")
        title = box.get("title") or f"Box {index}"

        x1 = int(round(box["x"] * img_w))
        y1 = int(round(box["y"] * img_h))
        x2 = int(round((box["x"] + box["w"]) * img_w))
        y2 = int(round((box["y"] + box["h"]) * img_h))

        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(x1 + 5, min(img_w, x2))
        y2 = max(y1 + 5, min(img_h, y2))

        crop = image[y1:y2, x1:x2]
        crop_filename = f"{index:02d}_{slot}.jpg"
        crop_path = os.path.join(this_crop_dir, crop_filename)
        cv2.imwrite(crop_path, crop)

        # Debug nội bộ để kiểm tra crop; frontend không bắt buộc hiển thị ảnh debug này.
        cv2.rectangle(debug, (x1, y1), (x2, y2), (30, 136, 255), max(2, img_w // 450))
        cv2.putText(debug, title, (x1 + 8, max(28, y1 + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 136, 255), 2)

        crops.append({
            "slot": slot,
            "title": title,
            "image": crop,
            "crop_path": crop_path,
            "crop_url": f"/uploads/crops/{request_id}/{crop_filename}",
        })

    debug_path = os.path.join(DEBUG_DIR, f"{request_id}_boxes.jpg")
    cv2.imwrite(debug_path, debug)
    return crops


def pad_to_square_bgr(img):
    """Padding ảnh thành hình vuông để resize không làm méo món."""
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)


def reduce_glare_bgr(img):
    """Tiền xử lý nhẹ để giảm ảnh hưởng chói sáng nhưng không làm đổi màu quá mạnh."""
    if not APPLY_GLARE_PREPROCESSING or img is None or img.size == 0:
        return img
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    except Exception:
        return img


def preprocess_for_cnn(crop_img):
    """BGR image -> RGB tensor 1x128x128x3, scale 0..1."""
    if crop_img is None or crop_img.size == 0:
        raise ValueError("Ảnh crop rỗng")
    img = pad_to_square_bgr(crop_img)
    img = reduce_glare_bgr(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE_CNN, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_dish_cnn1(crop_img) -> Tuple[str, float]:
    if cnn1_model is None:
        raise RuntimeError("CNN món ăn chưa load. Kiểm tra models/model_food.h5 và tensorflow.")
    if not cnn1_classes:
        raise RuntimeError("CNN món ăn chưa có class. Kiểm tra models/class_indices.json.")

    img = preprocess_for_cnn(crop_img)
    pred = cnn1_model.predict(img, verbose=0)
    idx = int(np.argmax(pred[0]))
    if idx >= len(cnn1_classes):
        raise RuntimeError(f"Model trả class index {idx}, nhưng class_indices chỉ có {len(cnn1_classes)} class")
    label = cnn1_classes[idx]
    conf = float(pred[0][idx])
    return safe_label(label), conf


def predict_egg_cell(cell_img) -> Tuple[bool, float]:
    """Trả về crop cell có phải trứng không. Class 1 = Trung-trong-thit-kho."""
    if egg_cnn_model is None:
        return False, 0.0
    img = preprocess_for_cnn(cell_img)
    pred = egg_cnn_model.predict(img, verbose=0)
    egg_conf = float(pred[0][1]) if pred.shape[-1] >= 2 else float(pred[0][0])
    return egg_conf >= EGG_CONFIDENCE_THRESHOLD, egg_conf


def count_components(binary_grid: List[List[bool]]) -> int:
    """Đếm cụm ô dương tính liền kề để tránh đếm trùng một quả trứng."""
    rows = len(binary_grid)
    cols = len(binary_grid[0]) if rows else 0
    visited = [[False] * cols for _ in range(rows)]
    components = 0
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for r in range(rows):
        for c in range(cols):
            if not binary_grid[r][c] or visited[r][c]:
                continue
            components += 1
            stack = [(r, c)]
            visited[r][c] = True
            while stack:
                cr, cc = stack.pop()
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and binary_grid[nr][nc] and not visited[nr][nc]:
                        visited[nr][nc] = True
                        stack.append((nr, nc))
    return components


def count_eggs_with_grid_cnn(dish_img) -> Tuple[int, str]:
    """
    Không dùng YOLO. Chia box thịt kho thành 5x6 = 30 ô không trùng nhau,
    dùng CNN phụ kiểm tra từng ô, gom các ô trứng liền kề thành 1 cụm.
    """
    if egg_cnn_model is None:
        return 0, "CNN trứng chưa load"

    h, w = dish_img.shape[:2]
    binary = [[False for _ in range(EGG_GRID_COLS)] for _ in range(EGG_GRID_ROWS)]
    positive_cells = []

    for r in range(EGG_GRID_ROWS):
        y1 = int(round(r * h / EGG_GRID_ROWS))
        y2 = int(round((r + 1) * h / EGG_GRID_ROWS))
        for c in range(EGG_GRID_COLS):
            x1 = int(round(c * w / EGG_GRID_COLS))
            x2 = int(round((c + 1) * w / EGG_GRID_COLS))
            cell = dish_img[y1:y2, x1:x2]
            is_egg, conf = predict_egg_cell(cell)
            if is_egg:
                binary[r][c] = True
                positive_cells.append((r, c, round(conf, 4)))

    egg_count = min(count_components(binary), MAX_EGGS)
    note = f"Grid {EGG_GRID_ROWS}x{EGG_GRID_COLS}, ô trứng: {len(positive_cells)}, cụm trứng: {egg_count}"
    return egg_count, note


def calculate_thit_kho_price(egg_count: int) -> Tuple[str, int, str]:
    if egg_count <= 0:
        return "Thịt kho", 25000, "Không có trứng"
    if egg_count == 1:
        return "Thịt kho trứng", 30000, "1 trứng"
    price = 30000 + (egg_count - 1) * 6000
    return f"Thịt kho trứng ({egg_count} trứng)", price, f"{egg_count} trứng"


def build_item_from_label(label: str, crop_img, conf: float, crop_url: str, slot: str, title: str = "") -> Optional[Dict]:
    label = safe_label(label)
    print(f"[PREDICT] {slot}: {label} | conf={conf:.4f}")

    if label == "thit_kho":
        egg_count, egg_debug_note = count_eggs_with_grid_cnn(crop_img)
        name, price, note = calculate_thit_kho_price(egg_count)
        if egg_debug_note:
            note = f"{note} | {egg_debug_note}"
    else:
        price = MENU_PRICE.get(label, 0)
        name = DISPLAY_NAME.get(label, label)
        note = ""
        if price == 0:
            note = "Chưa có giá / kiểm tra MENU_PRICE"

    return {
        "slot": slot,
        "title": title or slot,
        "name": name,
        "label": label,
        "quantity": 1,
        "price": int(price),
        "price_text": vnd(int(price)),
        "note": note,
        "confidence": round(float(conf), 4),
        "crop_url": crop_url,
    }


def predict_from_image_path(image_path: str, boxes: Optional[List[Dict]] = None) -> Dict:
    request_id = uuid.uuid4().hex[:10]
    if boxes is None:
        boxes = default_roi_boxes()
    crops = crop_custom_boxes(image_path, request_id, boxes)

    items = []
    for crop_data in crops:
        slot = crop_data["slot"]
        title = crop_data.get("title", slot)
        crop_img = crop_data["image"]
        crop_url = crop_data["crop_url"]

        label, conf = predict_dish_cnn1(crop_img)
        item = build_item_from_label(label, crop_img, conf, crop_url, slot, title)
        if item is not None:
            items.append(item)

    total = sum(item["price"] * item.get("quantity", 1) for item in items)

    return {
        "request_id": request_id,
        "items": items,
        "total": total,
        "total_text": vnd(total),
        "invoice_id": f"HD{datetime.now().strftime('%H%M%S')}",
        "status": "Chưa thanh toán",
        "debug_roi_url": f"/uploads/debug/{request_id}_boxes.jpg",
        "model_status": {
            "recognizer_ready": cnn1_model is not None and len(cnn1_classes) > 0,
            "cnn1_loaded": cnn1_model is not None,
            "cnn1_classes_loaded": len(cnn1_classes) > 0,
            "cnn1_classes_count": len(cnn1_classes),
            "cnn1_classes": cnn1_classes,
            "egg_cnn_loaded": egg_cnn_model is not None,
            "egg_cnn_classes_loaded": len(egg_cnn_classes) > 0,
            "egg_cnn_classes": egg_cnn_classes,
            "egg_grid": f"{EGG_GRID_ROWS}x{EGG_GRID_COLS}",
            "egg_confidence_threshold": EGG_CONFIDENCE_THRESHOLD,
            "max_eggs": MAX_EGGS,
            "uses_yolo": False,
        }
    }


def save_image_from_request() -> str:
    image_path = None
    if "image" in request.files:
        image_path = save_upload_file(request.files["image"])
    else:
        data = request.get_json(silent=True) or {}
        image_data = data.get("image_data")
        if image_data:
            image_path = save_base64_image(image_data)

    if not image_path:
        raise ValueError("Không có ảnh upload hoặc image_data")
    return image_path


# =========================
# ROUTES
# =========================
@app.route("/assets/<path:filename>")
def asset_file(filename):
    return send_from_directory(ASSET_DIR, filename)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/prepare_image", methods=["POST"])
def api_prepare_image():
    """
    Bước 1: lưu ảnh trước, trả ảnh + 5 box mặc định để người dùng chỉnh.
    Chưa detect ở bước này.
    """
    try:
        image_path = save_image_from_request()
        token = os.path.basename(image_path)
        return jsonify({
            "ok": True,
            "image_token": token,
            "original_url": "/uploads/originals/" + token,
            "boxes": default_roi_boxes(),
            "message": "Ảnh đã được lưu. Hãy chỉnh box rồi bấm nhận diện."
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/predict_boxes", methods=["POST"])
def api_predict_boxes():
    """
    Bước 2: nhận danh sách box đã chỉnh từ frontend -> crop -> detect -> trả hóa đơn.
    """
    try:
        data = request.get_json(silent=True) or {}
        image_token = data.get("image_token")
        boxes = data.get("boxes") or []
        image_path = get_original_path_from_token(image_token)
        result = predict_from_image_path(image_path, boxes=boxes)
        result["ok"] = True
        result["original_url"] = "/uploads/originals/" + normalize_token(image_token)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API cũ: upload/chụp xong detect luôn bằng 5 box mặc định.
    Giữ lại để tương thích, frontend mới không dùng route này làm luồng chính.
    """
    try:
        image_path = save_image_from_request()
        result = predict_from_image_path(image_path, boxes=default_roi_boxes())
        result["ok"] = True
        result["original_url"] = "/uploads/originals/" + os.path.basename(image_path)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


# =========================
# FRONTEND HTML
# =========================
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Checkie - Căn tin</title>
  <style>
    :root {
      --blue: #1e88ff;
      --blue-2: #0d6fe8;
      --blue-soft: #eaf5ff;
      --border: #cfe6ff;
      --text: #172033;
      --muted: #607d8b;
      --success: #13b85d;
      --danger: #e31b3c;
      --card: rgba(255,255,255,0.94);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      min-height: 100vh;
      color: var(--text);
      background-color: #f6fbff;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Roboto, "Helvetica Neue", Arial, sans-serif;
      text-rendering: geometricPrecision;
      -webkit-font-smoothing: antialiased;
      background-image:
        linear-gradient(45deg, rgba(30,136,255,0.06) 25%, transparent 25%),
        linear-gradient(-45deg, rgba(30,136,255,0.06) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, rgba(30,136,255,0.06) 75%),
        linear-gradient(-45deg, transparent 75%, rgba(30,136,255,0.06) 75%);
      background-size: 36px 36px;
      background-position: 0 0, 0 18px, 18px -18px, -18px 0px;
    }
    .container { max-width: 1450px; margin: auto; padding: 20px; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
    .logo { display: flex; align-items: center; gap: 12px; font-size: 42px; font-weight: 900; font-style: italic; color: var(--blue); letter-spacing: -1.4px; }
    .logo-icon { width: 56px; height: 56px; background: var(--blue); color: white; border-radius: 50%; display: grid; place-items: center; font-size: 34px; }
    .admin { display: flex; align-items: center; gap: 10px; font-weight: 750; }
    .avatar { width: 46px; height: 46px; border-radius: 50%; background: var(--blue-soft); border: 2px solid var(--border); display: grid; place-items: center; font-size: 24px; }

    .top-card {
      position: relative;
      background: var(--card);
      border: 2px solid var(--border);
      border-radius: 24px;
      padding: 28px 38px;
      display: grid;
      grid-template-columns: 280px 1fr 235px;
      gap: 28px;
      align-items: center;
      min-height: 500px;
      box-shadow: 0 12px 28px rgba(30,136,255,0.12);
    }
    .product-title { font-size: 25px; font-weight: 900; margin-bottom: 28px; }
    .meta { font-size: 18px; font-weight: 750; color: #6b7280; margin-bottom: 20px; }
    .meta strong { color: var(--blue); }
    .system-note { font-size: 13px; color: var(--muted); line-height: 1.55; background: #f8fcff; border: 1px solid var(--border); border-radius: 12px; padding: 12px; }

    .recognition-area { width: 100%; }
    .camera-box {
      width: 100%;
      max-width: 760px;
      min-height: 360px;
      margin: auto;
      border-radius: 16px;
      overflow: hidden;
      border: 2px solid #d5eaff;
      background: var(--blue-soft);
      box-shadow: 0 14px 30px rgba(0,0,0,0.14);
      position: relative;
      display: grid;
      place-items: center;
    }
    video { width: 100%; height: 100%; object-fit: cover; display: none; min-height: 360px; }
    canvas { display: none; }
    .placeholder { width: 100%; min-height: 360px; display: grid; place-items: center; text-align: center; color: var(--muted); font-weight: 750; padding: 28px; line-height: 1.5; }

    .editor-wrap {
      display: none;
      position: relative;
      width: 100%;
      background: #f8fcff;
    }
    .editor-wrap img {
      width: 100%;
      height: auto;
      display: block;
      user-select: none;
      -webkit-user-drag: none;
    }
    .roi-layer { position: absolute; inset: 0; touch-action: none; }
    .roi-box {
      position: absolute;
      border: 3px solid var(--blue);
      border-radius: 12px;
      background: rgba(30, 136, 255, 0.07);
      box-shadow: 0 0 0 2px rgba(255,255,255,0.8), 0 10px 20px rgba(0,0,0,0.18);
      cursor: move;
      min-width: 44px;
      min-height: 44px;
    }
    .roi-box.active { border-color: #ffb000; background: rgba(255, 176, 0, 0.10); }
    .roi-label {
      position: absolute;
      left: 8px;
      top: 8px;
      background: var(--blue);
      color: white;
      padding: 5px 9px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 900;
      line-height: 1;
      box-shadow: 0 4px 10px rgba(0,0,0,0.18);
      pointer-events: none;
    }
    .roi-delete {
      position: absolute;
      right: -12px;
      top: -12px;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      border: 2px solid white;
      background: var(--danger);
      color: white;
      font-size: 19px;
      font-weight: 900;
      display: grid;
      place-items: center;
      cursor: pointer;
      z-index: 3;
      box-shadow: 0 5px 12px rgba(0,0,0,0.22);
    }
    .roi-handle {
      position: absolute;
      right: -7px;
      bottom: -7px;
      width: 22px;
      height: 22px;
      border-radius: 8px;
      background: white;
      border: 3px solid var(--blue);
      cursor: nwse-resize;
      z-index: 2;
    }
    .roi-toolbar {
      display: none;
      margin-top: 14px;
      gap: 12px;
      justify-content: center;
      align-items: center;
      flex-wrap: wrap;
    }
    .roi-help {
      display: none;
      margin-top: 10px;
      color: var(--muted);
      text-align: center;
      font-size: 13px;
      font-weight: 700;
      line-height: 1.45;
    }
    .camera-actions { margin-top: 16px; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }
    button { cursor: pointer; font-family: inherit; }
    .btn { border: none; border-radius: 12px; padding: 12px 18px; font-weight: 900; font-size: 15px; transition: 0.15s ease; }
    .btn:hover { transform: translateY(-1px); }
    .btn-primary { background: var(--blue); color: white; box-shadow: 0 10px 18px rgba(30,136,255,0.22); }
    .btn-light { background: white; color: var(--blue); border: 1px solid var(--border); }
    .btn-danger { background: #fff1f3; color: var(--danger); border: 1px solid #ffd2da; }
    .btn:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }

    .status-pill { position: absolute; top: 46px; right: 38px; background: #f4fbff; border: 2px solid var(--border); color: var(--blue); border-radius: 10px; padding: 10px 18px; font-weight: 900; }
    .dot { width: 11px; height: 11px; background: var(--success); display: inline-block; border-radius: 50%; margin-right: 8px; }
    .heart { position: absolute; right: 92px; bottom: 70px; font-size: 78px; color: rgba(30,136,255,0.22); transform: rotate(10deg); }

    .bottom-card {
      margin-top: 10px;
      background: var(--card);
      border: 2px solid var(--border);
      border-radius: 24px;
      padding: 28px 38px;
      display: grid;
      grid-template-columns: 1.1fr 1fr 1.1fr;
      gap: 32px;
      box-shadow: 0 12px 28px rgba(30,136,255,0.12);
    }
    .section { min-height: 510px; position: relative; }
    .section:not(:last-child) { border-right: 2px solid var(--border); padding-right: 32px; }
    .section-title { color: var(--blue); font-size: 21px; font-weight: 900; margin-bottom: 24px; }
    .item-row { display: grid; grid-template-columns: 1fr 45px 110px; align-items: center; margin-bottom: 18px; font-size: 17px; }
    .item-name { font-weight: 800; }
    .qty { color: #6b7280; font-weight: 750; }
    .price { text-align: right; color: var(--blue); font-weight: 900; }
    .small { color: var(--muted); font-size: 12px; margin-top: 4px; line-height: 1.4; }
    .crop-grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:10px; margin-top:15px; max-height: 250px; overflow:auto; }
    .crop-card { border:1px solid var(--border); border-radius:12px; padding:8px; background:#f8fcff; }
    .crop-card img { width:100%; border-radius:8px; display:block; }

    .qr-card {
      width: 320px;
      max-width: 100%;
      margin: auto;
      background: white;
      border: 2px solid var(--border);
      border-radius: 20px;
      padding: 12px;
      color: var(--text);
      text-align: center;
      box-shadow: 0 14px 28px rgba(30,136,255,0.18);
    }
    .qr-image { width: 100%; height: auto; display: block; border-radius: 16px; }
    .qr-caption { margin-top: 10px; color: var(--muted); font-size: 13px; font-weight: 800; }
    .print-btn { position: absolute; bottom: 0; left: 50%; transform: translateX(-50%); width: 80%; border: 2px solid var(--border); color: var(--blue); background: white; padding: 14px; border-radius: 12px; font-weight: 900; font-size: 16px; }
    .summary-row { display: flex; justify-content: space-between; font-size: 18px; font-weight: 750; margin-bottom: 28px; }
    .blue { color: var(--blue); font-weight: 900; }
    .red { color: red; font-weight: 900; }
    .divider { border-top: 2px dashed var(--border); margin: 18px 0 36px; }
    .final-row { display: flex; justify-content: space-between; align-items: center; font-size: 19px; font-weight: 800; margin-bottom: 28px; }
    .final-money { font-size: 34px; color: var(--blue); font-weight: 950; }
    .money-input { width: 190px; padding: 14px; border-radius: 12px; border: 2px solid var(--border); text-align: right; font-weight: 800; font-size: 18px; }
    .change-box { width: 190px; padding: 14px; border-radius: 12px; background: #f0fff4; border: 2px solid #d7f3df; text-align: right; color: #00943a; font-weight: 900; font-size: 18px; }
    .confirm-btn { width: 100%; margin-top: 24px; padding: 18px; border: none; border-radius: 12px; background: var(--blue); color: white; font-size: 20px; font-weight: 950; }
    .thanks { text-align: center; margin-top: 28px; color: var(--blue); font-size: 19px; font-weight: 800; font-style: italic; }
    .success { display: none; text-align: center; margin-top: 12px; color: var(--success); font-weight: 900; }

    .print-receipt { display: none; }
    .receipt-title { text-align: center; font-size: 22px; font-weight: 950; margin-bottom: 18px; }
    .receipt-items { border-top: 1px solid #111; border-bottom: 1px solid #111; padding: 12px 0; margin: 12px 0; }
    .receipt-row { display: flex; justify-content: space-between; gap: 18px; font-size: 15px; line-height: 1.45; margin: 8px 0; }
    .receipt-row span:first-child { flex: 1; }
    .receipt-row span:last-child { white-space: nowrap; font-weight: 800; }
    .receipt-total { display: flex; justify-content: space-between; gap: 18px; font-size: 18px; font-weight: 950; margin-top: 14px; }

    @media print {
      @page { margin: 12mm; }
      body { background: white !important; color: #111 !important; }
      .container { display: none !important; }
      .print-receipt {
        display: block !important;
        width: 100%;
        max-width: 420px;
        margin: 0 auto;
        padding: 0;
        background: white;
        color: #111;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Arial, sans-serif;
      }
    }

    @media (max-width: 1050px) {
      .top-card, .bottom-card { grid-template-columns: 1fr; }
      .section:not(:last-child) { border-right: none; border-bottom: 2px solid var(--border); padding-right: 0; padding-bottom: 30px; }
      .print-btn { position: static; transform: none; width: 100%; margin-top: 20px; }
      .status-pill, .heart { display: none; }
    }
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="logo"><div class="logo-icon">✓</div> checkie ✦</div>
    <div class="admin"><div class="avatar">NV</div><span>Nhân viên:</span><strong>Admin⌄</strong></div>
  </div>

  <div class="top-card">
    <div>
      <div class="product-title">SUẤT CƠM VĂN PHÒNG</div>
      <div class="meta">Mã: <strong id="invoiceCode">HD000</strong></div>
      <div class="meta">Giá: <strong id="comboPrice">0 đ</strong></div>
      <div class="system-note" id="systemNote">Tải ảnh hoặc chụp ảnh khay. Sau đó chỉnh box cho khớp từng món rồi bấm nhận diện.</div>
    </div>

    <div class="recognition-area">
      <div class="camera-box" id="cameraBox">
        <video id="camera" autoplay playsinline muted></video>
        <canvas id="canvas"></canvas>
        <div id="editorWrap" class="editor-wrap">
          <img id="trayImage" alt="Ảnh khay đã lưu" />
          <div id="roiLayer" class="roi-layer"></div>
        </div>
        <div id="placeholder" class="placeholder">
          Chỉ dùng 2 thao tác chính: Tải ảnh lên hoặc Chụp ảnh.<br>
          Ảnh sẽ được lưu trước, sau đó bạn có thể chỉnh 5 box nhận diện trên khay.
        </div>
      </div>

      <div class="camera-actions">
        <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="uploadFile()">
        <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">Tải ảnh lên</button>
        <button class="btn btn-light" id="captureBtn" onclick="handleCaptureButton()">Chụp ảnh</button>
      </div>

      <div class="roi-toolbar" id="roiToolbar">
        <button class="btn btn-primary" id="predictBtn" onclick="predictCurrentBoxes()">Nhận diện món trong box</button>
        <button class="btn btn-light" onclick="restoreDefaultBoxes()">Khôi phục 5 box</button>
        <button class="btn btn-danger" onclick="clearAllBoxes()">Xóa tất cả box</button>
      </div>
      <div class="roi-help" id="roiHelp">
        Kéo thân box để đổi vị trí, kéo góc dưới phải để đổi kích thước, bấm dấu × để xóa box nếu khay chỉ có 4 món.
      </div>
    </div>

    <div><div class="status-pill"><span class="dot"></span>Sẵn sàng</div><div class="heart">💙</div></div>
  </div>

  <div class="bottom-card">
    <div class="section">
      <div class="section-title">MÓN ĐÃ CHỌN</div>
      <div id="itemsList"><div class="small">Chưa có kết quả. Hãy tải ảnh hoặc chụp ảnh để tạo box trước.</div></div>
      <div class="crop-grid" id="cropGrid"></div>
    </div>

    <div class="section">
      <div class="section-title">THANH TOÁN</div>
      <div class="qr-card">
        <img class="qr-image" src="/assets/vietcombank_qr.jpg" alt="Vietcombank QR thanh toán">
        <div class="qr-caption">Quét mã Vietcombank để thanh toán</div>
        <span id="qrContent" style="display:none;">HD000 - Admin</span>
      </div>
      <button class="print-btn" onclick="window.print()">In hóa đơn</button>
    </div>

    <div class="section">
      <div class="section-title">TỔNG THANH TOÁN</div>
      <div class="summary-row"><span>Tạm tính (<span id="itemCount">0</span> món)</span><span class="blue" id="subtotalText">0 đ</span></div>
      <div class="summary-row"><span>Giảm giá</span><span class="red">0 đ</span></div>
      <div class="divider"></div>
      <div class="final-row"><span>Thành tiền</span><span class="final-money" id="finalText">0 đ</span></div>
      <div class="final-row"><span>Khách đưa</span><input id="customerPaid" class="money-input" type="number" value="0" min="0" oninput="updateChange()"></div>
      <div class="final-row"><span>Tiền thừa</span><div class="change-box" id="changeText">0 đ</div></div>
      <button class="confirm-btn" onclick="confirmPayment()">✓ XÁC NHẬN THANH TOÁN</button>
      <div id="successMessage" class="success">Thanh toán thành công!</div>
      <div class="thanks">"Cảm ơn quý khách và hẹn gặp lại!"</div>
    </div>
  </div>
</div>

<div id="printReceipt" class="print-receipt">
  <div class="receipt-title">HÓA ĐƠN THANH TOÁN</div>
  <div id="printItems" class="receipt-items">Chưa có món được chọn.</div>
  <div class="receipt-total">
    <span>Tổng tiền</span>
    <span id="printTotalText">0 đ</span>
  </div>
</div>

<script>
let stream = null;
let currentTotal = 0;
let currentImageToken = null;
let boxes = [];
let defaultBoxes = [];
let activeBoxId = null;
let interaction = null;

function formatVND(number) { return Number(number || 0).toLocaleString('vi-VN') + ' đ'; }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function uid() { return 'box_' + Math.random().toString(16).slice(2, 10); }

function updateChange() {
  const paid = Number(document.getElementById('customerPaid').value || 0);
  const change = Math.max(0, paid - currentTotal);
  document.getElementById('changeText').textContent = formatVND(change);
}
function confirmPayment() { document.getElementById('successMessage').style.display = 'block'; }
function setNote(text) { document.getElementById('systemNote').textContent = text; }

function clearInvoice() {
  currentTotal = 0;
  document.getElementById('invoiceCode').textContent = 'HD000';
  document.getElementById('qrContent').textContent = 'HD000 - Admin';
  document.getElementById('comboPrice').textContent = '0 đ';
  document.getElementById('subtotalText').textContent = '0 đ';
  document.getElementById('finalText').textContent = '0 đ';
  document.getElementById('itemCount').textContent = '0';
  document.getElementById('itemsList').innerHTML = '<div class="small">Ảnh đã lưu. Hãy chỉnh box rồi bấm nhận diện món trong box.</div>';
  document.getElementById('cropGrid').innerHTML = '';
  document.getElementById('successMessage').style.display = 'none';
  document.getElementById('printItems').innerHTML = 'Chưa có món được chọn.';
  document.getElementById('printTotalText').textContent = '0 đ';
  updateChange();
}

function renderResult(data) {
  if (!data.ok) { alert(data.error || 'Có lỗi xảy ra'); return; }
  currentTotal = data.total || 0;
  document.getElementById('invoiceCode').textContent = data.invoice_id || 'HD000';
  document.getElementById('qrContent').textContent = (data.invoice_id || 'HD000') + ' - Admin';
  document.getElementById('comboPrice').textContent = data.total_text || '0 đ';
  document.getElementById('subtotalText').textContent = data.total_text || '0 đ';
  document.getElementById('finalText').textContent = data.total_text || '0 đ';
  document.getElementById('itemCount').textContent = (data.items || []).length;

  const list = document.getElementById('itemsList');
  list.innerHTML = '';
  if (!data.items || data.items.length === 0) {
    list.innerHTML = '<div class="small">Không có món nào được nhận diện. Hãy kiểm tra lại box.</div>';
  } else {
    data.items.forEach(item => {
      const noteText = item.note ? `<div class="small">${item.title || item.slot} | ${item.note}</div>` : `<div class="small">${item.title || item.slot}</div>`;
      list.innerHTML += `
        <div class="item-row">
          <div><div class="item-name">${item.name}</div>${noteText}</div>
          <div class="qty">× ${item.quantity}</div>
          <div class="price">${item.price_text}</div>
        </div>`;
    });
  }

  const cropGrid = document.getElementById('cropGrid');
  cropGrid.innerHTML = '';
  (data.items || []).forEach(item => {
    cropGrid.innerHTML += `
      <div class="crop-card">
        <img src="${item.crop_url}" />
        <div class="small">${item.title || item.slot}: ${item.name}</div>
      </div>`;
  });

  const printItems = document.getElementById('printItems');
  printItems.innerHTML = '';
  if (!data.items || data.items.length === 0) {
    printItems.textContent = 'Chưa có món được chọn.';
  } else {
    data.items.forEach(item => {
      printItems.innerHTML += `
        <div class="receipt-row">
          <span>${item.name} × ${item.quantity}</span>
          <span>${item.price_text}</span>
        </div>`;
    });
  }
  document.getElementById('printTotalText').textContent = data.total_text || '0 đ';

  updateChange();
  setNote('Đã nhận diện xong. Bạn vẫn có thể chỉnh box và nhận diện lại nếu kết quả chưa đúng.');
}

function showPreparedImage(data) {
  if (!data.ok) { alert(data.error || 'Có lỗi xảy ra'); return; }
  currentImageToken = data.image_token;
  defaultBoxes = JSON.parse(JSON.stringify(data.boxes || []));
  boxes = (data.boxes || []).map((b, idx) => ({
    id: b.id || uid(),
    slot: b.slot || ('box_' + (idx + 1)),
    title: b.title || ('Box ' + (idx + 1)),
    x: Number(b.x), y: Number(b.y), w: Number(b.w), h: Number(b.h),
    deleted: false
  }));
  clearInvoice();

  const video = document.getElementById('camera');
  const placeholder = document.getElementById('placeholder');
  const editorWrap = document.getElementById('editorWrap');
  const trayImage = document.getElementById('trayImage');
  const roiToolbar = document.getElementById('roiToolbar');
  const roiHelp = document.getElementById('roiHelp');

  video.style.display = 'none';
  placeholder.style.display = 'none';
  trayImage.onload = () => {
    editorWrap.style.display = 'block';
    roiToolbar.style.display = 'flex';
    roiHelp.style.display = 'block';
    renderBoxes();
  };
  trayImage.src = data.original_url + '?t=' + Date.now();
  setNote('Ảnh đã được lưu. Hãy chỉnh box cho khớp từng món rồi bấm nhận diện.');
}

function renderBoxes() {
  const layer = document.getElementById('roiLayer');
  layer.innerHTML = '';
  boxes.filter(b => !b.deleted).forEach((box, visibleIndex) => {
    const div = document.createElement('div');
    div.className = 'roi-box' + (box.id === activeBoxId ? ' active' : '');
    div.dataset.id = box.id;
    div.style.left = (box.x * 100) + '%';
    div.style.top = (box.y * 100) + '%';
    div.style.width = (box.w * 100) + '%';
    div.style.height = (box.h * 100) + '%';

    const label = document.createElement('div');
    label.className = 'roi-label';
    label.textContent = box.title || ('Box ' + (visibleIndex + 1));

    const del = document.createElement('button');
    del.className = 'roi-delete';
    del.type = 'button';
    del.textContent = '×';
    del.title = 'Xóa box này';
    del.addEventListener('pointerdown', (e) => e.stopPropagation());
    del.addEventListener('click', (e) => {
      e.stopPropagation();
      box.deleted = true;
      if (activeBoxId === box.id) activeBoxId = null;
      renderBoxes();
      setNote('Đã xóa box. Khi khay chỉ có 4 món, chỉ giữ lại 4 box tương ứng.');
    });

    const handle = document.createElement('div');
    handle.className = 'roi-handle';
    handle.title = 'Kéo để đổi kích thước';

    div.appendChild(label);
    div.appendChild(del);
    div.appendChild(handle);
    div.addEventListener('pointerdown', startBoxInteraction);
    layer.appendChild(div);
  });
}

function updateBoxElement(id) {
  const box = boxes.find(b => b.id === id);
  const el = document.querySelector(`.roi-box[data-id="${id}"]`);
  if (!box || !el) return;
  el.style.left = (box.x * 100) + '%';
  el.style.top = (box.y * 100) + '%';
  el.style.width = (box.w * 100) + '%';
  el.style.height = (box.h * 100) + '%';
}

function startBoxInteraction(event) {
  const target = event.target;
  if (target.classList.contains('roi-delete')) return;
  const boxEl = event.currentTarget;
  const id = boxEl.dataset.id;
  const box = boxes.find(b => b.id === id);
  if (!box) return;
  activeBoxId = id;
  renderBoxes();

  const layer = document.getElementById('roiLayer');
  const rect = layer.getBoundingClientRect();
  interaction = {
    id,
    type: target.classList.contains('roi-handle') ? 'resize' : 'move',
    startX: event.clientX,
    startY: event.clientY,
    layerW: rect.width,
    layerH: rect.height,
    startBox: { x: box.x, y: box.y, w: box.w, h: box.h }
  };
  event.preventDefault();
}

document.addEventListener('pointermove', (event) => {
  if (!interaction) return;
  const box = boxes.find(b => b.id === interaction.id);
  if (!box) return;
  const dx = (event.clientX - interaction.startX) / interaction.layerW;
  const dy = (event.clientY - interaction.startY) / interaction.layerH;

  if (interaction.type === 'move') {
    box.x = clamp(interaction.startBox.x + dx, 0, 1 - box.w);
    box.y = clamp(interaction.startBox.y + dy, 0, 1 - box.h);
  } else {
    box.w = clamp(interaction.startBox.w + dx, 0.05, 1 - box.x);
    box.h = clamp(interaction.startBox.h + dy, 0.05, 1 - box.y);
  }
  updateBoxElement(box.id);
});

document.addEventListener('pointerup', () => { interaction = null; });

function restoreDefaultBoxes() {
  boxes = defaultBoxes.map((b, idx) => ({
    id: b.id || uid(),
    slot: b.slot || ('box_' + (idx + 1)),
    title: b.title || ('Box ' + (idx + 1)),
    x: Number(b.x), y: Number(b.y), w: Number(b.w), h: Number(b.h),
    deleted: false
  }));
  activeBoxId = null;
  renderBoxes();
  setNote('Đã khôi phục 5 box mặc định.');
}

function clearAllBoxes() {
  boxes.forEach(b => b.deleted = true);
  activeBoxId = null;
  renderBoxes();
  setNote('Đã xóa tất cả box. Hãy khôi phục 5 box nếu muốn chỉnh lại.');
}

async function uploadFile() {
  const input = document.getElementById('fileInput');
  if (!input.files.length) return;
  const formData = new FormData();
  formData.append('image', input.files[0]);
  setNote('Đang lưu ảnh...');

  try {
    const res = await fetch('/api/prepare_image', { method: 'POST', body: formData });
    const data = await res.json();
    showPreparedImage(data);
  } catch (err) {
    console.error(err);
    alert('Không lưu được ảnh. Kiểm tra server Flask.');
    setNote('Lỗi khi lưu ảnh.');
  } finally {
    input.value = '';
  }
}

async function startPreferredCamera() {
  const video = document.getElementById('camera');
  const placeholder = document.getElementById('placeholder');
  const editorWrap = document.getElementById('editorWrap');
  const roiToolbar = document.getElementById('roiToolbar');
  const roiHelp = document.getElementById('roiHelp');

  try {
    setNote('Đang mở camera sau/camera ngoài nếu có...');
    if (stream) stopCamera(false);

    let tempStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });

    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(device => device.kind === 'videoinput');
    const preferred = choosePreferredCamera(cameras);
    tempStream.getTracks().forEach(track => track.stop());

    const videoConstraints = preferred
      ? { deviceId: { exact: preferred.deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } };

    stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
    video.srcObject = stream;
    video.style.display = 'block';
    placeholder.style.display = 'none';
    editorWrap.style.display = 'none';
    roiToolbar.style.display = 'none';
    roiHelp.style.display = 'none';
    await video.play();
    setNote('Camera đã bật. Canh khay vào đúng vị trí rồi bấm Chụp ảnh lần nữa để lưu ảnh.');
  } catch (err) {
    console.error(err);
    alert('Không mở được camera. Hãy chạy bằng localhost và cấp quyền camera cho trình duyệt.');
    setNote('Không mở được camera.');
  }
}

function choosePreferredCamera(cameras) {
  if (!cameras || cameras.length === 0) return null;
  const positive = /(back|rear|environment|world|sau|usb|external|hd|logitech)/i;
  const negative = /(front|user|facetime|integrated|built\s?in|trước)/i;
  let found = cameras.find(cam => positive.test(cam.label || ''));
  if (found) return found;
  found = cameras.find(cam => !negative.test(cam.label || ''));
  return found || cameras[cameras.length - 1];
}

function stopCamera(showPlaceholder = true) {
  const video = document.getElementById('camera');
  const placeholder = document.getElementById('placeholder');
  if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
  video.srcObject = null;
  video.style.display = 'none';
  if (showPlaceholder) placeholder.style.display = 'grid';
}

async function handleCaptureButton() {
  if (!stream) {
    await startPreferredCamera();
    return;
  }
  await captureImageForEditing();
}

async function captureImageForEditing() {
  const video = document.getElementById('camera');
  const canvas = document.getElementById('canvas');
  if (!stream || !video.videoWidth) { alert('Camera chưa sẵn sàng.'); return; }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL('image/png');
  setNote('Đang lưu ảnh đã chụp...');

  try {
    const res = await fetch('/api/prepare_image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_data: imageData })
    });
    const data = await res.json();
    stopCamera(false);
    showPreparedImage(data);
  } catch (err) {
    console.error(err);
    alert('Không lưu được ảnh đã chụp.');
    setNote('Lỗi khi lưu ảnh đã chụp.');
  }
}

async function predictCurrentBoxes() {
  if (!currentImageToken) {
    alert('Bạn cần tải ảnh hoặc chụp ảnh trước.');
    return;
  }
  const activeBoxes = boxes.filter(b => !b.deleted).map((b, idx) => ({
    id: b.id,
    slot: b.slot || ('box_' + (idx + 1)),
    title: b.title || ('Box ' + (idx + 1)),
    x: Number(b.x),
    y: Number(b.y),
    w: Number(b.w),
    h: Number(b.h)
  }));
  if (!activeBoxes.length) {
    alert('Không còn box nào để nhận diện. Hãy khôi phục 5 box hoặc tải/chụp ảnh lại.');
    return;
  }

  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  btn.textContent = 'Đang nhận diện...';
  setNote('Đang nhận diện các món trong box đã chọn...');

  try {
    const res = await fetch('/api/predict_boxes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_token: currentImageToken, boxes: activeBoxes })
    });
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    console.error(err);
    alert('Không nhận diện được. Kiểm tra server Flask/model.');
    setNote('Lỗi khi nhận diện.');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Nhận diện món trong box';
  }
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("=================================")
    print("Checkie canteen checkout v7 app is running")
    print("Open: http://127.0.0.1:5000")
    print("Model folder:", MODEL_DIR)
    print("Uploads folder:", UPLOAD_DIR)
    print("Press Ctrl + C to stop")
    print("=================================")
    app.run(debug=True, port=5000)
