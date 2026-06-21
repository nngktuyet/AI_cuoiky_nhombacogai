CHECKIE - WEB CHECKOUT CĂN TIN

File chính:
- checkie_backend_app_v7_qr_print.py

Logic backend hiện tại:
1. Giữ nguyên giao diện cũ: tải ảnh/chụp ảnh, chỉnh tối đa 5 box, nhận diện, QR cố định, in hóa đơn.
2. CNN chính dùng models/model_food.h5 để nhận diện 14 class món ăn, input 128x128x3.
3. Không dùng YOLO cá. Canh chua có cá / không cá được nhận diện trực tiếp bằng CNN chính.
4. Không dùng YOLO trứng và không còn file .pt trong thư mục models.
5. Nếu CNN chính dự đoán thit-kho:
   - Backend chia vùng thịt kho thành lưới 5x6 = 30 ô không trùng nhau.
   - Mỗi ô được đưa vào models/model_thitkho1.h5.
   - Class 1 với confidence >= 80% được xem là ô có trứng.
   - Các ô trứng liền kề được gom thành 1 cụm để tránh đếm trùng.
   - Tối đa 5 trứng.
6. Công thức thịt kho:
   - 0 trứng: Thịt kho = 25.000 đ
   - 1 trứng: Thịt kho trứng = 30.000 đ
   - Từ trứng thứ 2: +6.000 đ/trứng

Cấu trúc thư mục chính:
- checkie_backend_app_v7_qr_print.py
- assets/vietcombank_qr.jpg
- models/model_food.h5
- models/model_thitkho1.h5
- models/class_indices.json
- models/egg_class_indices.json
- uploads/originals: lưu ảnh upload/chụp
- uploads/crops: lưu ảnh crop theo box
- uploads/debug: lưu ảnh debug nội bộ
- requirements.txt
- start_checkie_windows.bat

Cách chạy nhanh trên Windows:
1. Giải nén file zip.
2. Mở thư mục vừa giải nén.
3. Double-click file:
   start_checkie_windows.bat
4. Sau khi server chạy, mở trình duyệt:
   http://127.0.0.1:5000

Cách chạy thủ công bằng terminal/cmd:
1. Mở terminal/cmd tại thư mục vừa giải nén.
2. Chạy lần lượt:
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python checkie_backend_app_v7_qr_print.py
3. Mở trình duyệt:
   http://127.0.0.1:5000

Lưu ý:
- Camera trên trình duyệt hoạt động tốt nhất khi chạy bằng localhost hoặc HTTPS.
- Nút "Chụp ảnh" có 2 bước: bấm lần 1 mở camera, bấm lần 2 chụp và lưu ảnh để chỉnh box.
- Nếu model không load được, hãy kiểm tra TensorFlow/Keras và file trong thư mục models.
- File start_checkie_windows.bat vẫn cài requirements khi mở để tránh thiếu thư viện trên máy mới.
