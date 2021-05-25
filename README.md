Camera Calibration by Windrist
=
Chương trình sử dụng thuật toán Camera Calibration và Detect Corners ứng dụng trong việc tìm khoảng cách ngoài thực tế giữa 2 điểm ảnh!

## Core Features:

```bash
"captureImage.py": Chụp ảnh bằng camera kết nối với máy tính
"getData.py": Tính toán các ma trận Extrinsic và Intrinsic và lưu vào thư mục 'output'
"result.py": Trả về khoảng cách 2 điểm ảnh đơn vị mm
```

## Requirements
- Python 3
- OpenCV (Python)
- Numpy

## Installation
```bash
git clone https://github.com/Windrist/Camera-Calibration
```

## Usage

- ### Demo:
```bash
cd Camera-Calibration
python getData.py
python result.py
```
- ### Manual:

```bash
cd Camera-Calibration
python captureImage.py
python getData.py
python result.py
```

Lưu ý:
- Cần sửa các tham số đã được liệt kê ở đầu các file code
- Muốn lưu ảnh chụp từ "captureImage" thì sử dụng phím "s", muốn thoát thì sử dụng phím "q"
- Muốn "result" trả về kết quả thì phải chọn 2 điểm trên ảnh

## ENJOY!
