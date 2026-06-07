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

Checkpoint TTS: hynt/ZipVoice-Vietnamese-2500h — Nguyen Thien Hy (hynt)
  https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h
ZipVoice gốc: https://github.com/k2-fsa/ZipVoice
Vocos ONNX (bundled): https://huggingface.co/wetdog/vocos-mel-24khz-onnx
Giấy phép weights: models/THIRD_PARTY_LICENSES.md

Chuẩn hóa text (GUI): pipeline tuần tự — VieNeu, Cấu trúc TTS, sea-g2p.
Nguồn GitHub — xem README.md / README_EN.md:
  VieNeu (port)     https://github.com/pnnbao97/VieNeu-TTS
  Cấu trúc TTS      period_linebreak.py — mã trong repo này
  sea-g2p           https://github.com/pnnbao97/sea-g2p

LICENSE: chỉ nghiên cứu / giáo dục / phi lợi nhuận — cấm thương mại.

