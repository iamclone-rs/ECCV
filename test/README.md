# Chạy SECO-SBIR trên QuickDraw bằng Kaggle

Split của lần chạy này lấy **đúng `UNSEEN_CLASSES["quickdraw"]` trong
`src/data_config.py` theo danh sách bạn cung cấp**. Không thay lớp test tự động.
Tên có dấu cách/gạch dưới được đối chiếu với tên thư mục thực tế; không đổi tên dữ liệu.

## Tham số

Tham khảo [SECO-SBIR, mục 4.7 và 5](https://arxiv.org/html/2608.03120v1):

| Tham số | Giá trị | Cơ sở |
| --- | --- | --- |
| Dataset | QuickDraw-Ext: 80 train / 30 test | Bài báo; danh sách lớp dùng data_config bạn cung cấp |
| Backbone | CLIP ViT-B/32 | Bài báo |
| Kích thước ảnh | 224 × 224 | Bài báo |
| Optimizer / LR | SGD / 2e-5 | Bài báo |
| Batch train / epochs | 32 / 10 | Bài báo |
| Triplet margin | 0.3 | Bài báo |
| lambda_class / lambda_consistency | 1 / 1 | Bài báo |
| Momentum / weight decay | 0.9 / 0.001 | Kế thừa mã dự án; bài chỉ nói SGD có momentum |
| Scheduler | StepLR, step_size=5, gamma=0.1 | Kế thừa mã dự án; bài không nêu |
| n_ctx / img_ctx / prompt_depth | 2 / 2 / 12 | Kế thừa mã dự án; không được nêu rõ trong bài |
| Temperature | 0.07 | Kế thừa mã dự án; bài không cho giá trị cụ thể |
| Batch trích xuất khi test | 128 | Chọn để giảm bộ nhớ trên Kaggle; không đổi batch train |
| workers / seed | 2 / 42 | Cấu hình chạy |

Lệnh train chọn `--loss_protocol paper`: consistency InfoNCE một chiều theo
phương trình 10–11 và contrastive sketch/photo hai chiều theo phương trình 13–15.
`--loss_protocol repo` giữ biến thể SimCLR vốn có trong repo. Các tham số cũ
`alpha`, `beta`, `gamma`, `lambd`, `distill` không điều khiển loss hiện tại.

Đây là cấu hình chạy của dự án dựa trên bài báo. Kiến trúc vẫn dùng chung
`adapter_photo` cho sketch/photo, residual adapter tỷ lệ 0.1 và cách khởi tạo
prompt của repo. Vì vậy không coi cấu hình là cam kết tái tạo chính xác con số
trong bài. Nhánh CLIP tham chiếu được đóng băng riêng để không bị cập nhật
LayerNorm từ nhánh text. Backbone chỉ mở LayerNorm, còn attention và positional
embeddings được đóng băng đúng mô tả bài báo. Giữ nguyên preprocessing của dự án.

## 1. Notebook online: tạo bundle

Bật Internet trong Kaggle. Sử dụng **mã nguồn đã sửa trong workspace này**:

- Cách A: commit/push thay đổi lên repo của bạn, rồi dán toàn bộ
  `test/kaggle_online.py` vào một cell. Mặc định clone
  `https://github.com/iamclone-rs/ECCV.git`, nhánh `main`. Có thể đặt `ECCV_REPO`
  và `ECCV_REF` trước khi chạy để chọn repo/nhánh khác.
- Cách B: upload thư mục dự án đã sửa lên Kaggle làm input, đặt
  `os.environ["ECCV_SOURCE"] = "/kaggle/input/<source-dataset>/ECCV"`, rồi chạy
  nội dung `test/kaggle_online.py`. Thư mục này phải chứa `src/`, `clip/`, `test/`.

Script đóng gói source thực tế, wheels ngoài CUDA/PyTorch và CLIP ViT-B/32,
kèm checksum và phiên bản môi trường. Không dùng các weights teacher DFN5B
trong mẫu KD-SBIR vì dự án này dùng CLIP làm nhánh tham chiếu.

Chọn **Save Version → Save & Run All**, lưu outputs. Bundle ở
`/kaggle/working/seco_quickdraw_<timestamp>/offline_bundle/`.

## 2. Notebook offline: khôi phục

Tạo notebook mới, bật GPU, tắt Internet. Thêm output notebook online và bộ
QuickDraw-Ext làm inputs. Dữ liệu cần là ảnh raster với cấu trúc:

```text
QuickDraw/
  sketch/<class_name>/*.png
  photo/<class_name>/*.jpg
```

QuickDraw Google 345 lớp chỉ có sketch không đáp ứng cấu trúc SBIR này.

Dán nội dung `test/kaggle_offline.py` vào cell đầu và chạy. Script kiểm tra
checksum, cài wheels offline, copy source vào `/kaggle/working/ECCV`, đặt CLIP
weights vào `/kaggle/working/clip_cache/ViT-B-32.pt`.

Nếu gắn nhiều bundle, đặt `ECCV_BUNDLE` trỏ tới đúng thư mục `offline_bundle`.
Online/offline phải dùng cùng Python và PyTorch image; script kiểm tra điều này.

## 3. Kiểm tra và train

Cell thứ hai dùng nội dung `test/kaggle_train.ipy`, với lệnh `!python -u -m
src.main_train` và đầy đủ tham số ngay trong cell. Bước kiểm tra tự tìm root chứa
`sketch/` và `photo/`, xác nhận đủ 80/30 lớp, lớp nào cũng có ảnh, và ghi
`/kaggle/working/quickdraw_preflight.json` trước khi khởi tạo model.

Nếu cần chọn đường dẫn thủ công:

```python
%env QUICKDRAW_ROOT=/kaggle/input/<dataset>/QuickDraw
%cd /kaggle/working/ECCV
!python test/kaggle_train.py --check_only
!python -u test/kaggle_train.py --smoke_test --exp_name seco_quickdraw_smoke
!python -u test/kaggle_train.py
```

Batch train mặc định giữ 32 theo bài. Nếu GPU hết bộ nhớ, thử giảm
`--test_batch_size 64` khi lỗi ở validation; nếu lỗi trong train, giảm
`--batch_size 16`. Batch train nhỏ hơn sẽ thay đổi tập negatives của contrastive
loss; tích lũy gradient không khôi phục được loss của batch 32.
Model hiện dùng weights half như repo gốc; tham số `prec` cũ chưa điều khiển AMP.
Chỉ sử dụng một GPU cho mỗi lần chạy.

## Output và resume

- `saved_models/seco_quickdraw/`: checkpoint tốt nhất theo mAP và `last.ckpt`.
- `saved_models/seco_quickdraw/run_config.json`: tham số, danh sách lớp, số ảnh.
- `csv_logs/` và `tb_logs/`: train loss, mAP@all và P@200.
- `seco_quickdraw.log`: output console khi chạy qua `!python test/kaggle_train.py`.
  Lệnh trực tiếp `!python -u -m src.main_train` hiển thị output trong notebook.

Validation dùng toàn bộ sketch và photo của 30 lớp test. Metric được tính theo
từng nhóm query để không tạo ma trận similarity cho toàn bộ dataset cùng lúc.

Lưu outputs sau mỗi phiên. Phiên tiếp theo, khôi phục bundle, gắn output phiên
trước rồi chạy:

```python
!python -u test/kaggle_train.py --epochs 10 --ckpt_path /kaggle/input/<previous-output>/saved_models/seco_quickdraw/last.ckpt
```

`epochs` là tổng số epoch đích. `--resume` của entrypoint khôi phục optimizer,
scheduler và epoch. `--ckpt_path` không có `--resume` chỉ nạp weights để fine-tune.
Không thay split, model hoặc loss giữa hai phiên resume.

Chạy regression checks từ root dự án:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
