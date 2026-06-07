Giọng mẫu được load từ ref_info.json + assets/ref_audio/*.wav

Đã kèm sẵn 9 giọng (xem bảng trong README.md):
  yen_nhi, my_van, ai_vy, an_nhi, dieu_linh,
  khanh_toan, tran_lam, nsnd_ha_phuong, nsnd_kim_cuc

Ví dụ assets/ref_info.json:

{
  "yen_nhi": {
    "name": "Yến Nhi",
    "audio_path": "assets/ref_audio/yen_nhi.wav",
    "text": "nội dung lời nói trong file wav..."   ← BẮT BUỘC (không auto transcribe)
  }
}

audio_path tìm theo thứ tự:
  assets/<path>
  <project_root>/<path>
  assets/ref_audio/<tên file>
  data/ref_audio/<tên file>

Sau khi sửa JSON, bấm "Làm mới danh sách" trên GUI.

LICENSE: chỉ nghiên cứu / giáo dục / phi lợi nhuận — cấm thương mại.
