Giọng mẫu được load theo HAI kiểu (gộp trong menu GUI):



1) ref_info.json + file audio theo audio_path

   Đã kèm sẵn 9 giọng:

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



2) Thư mục sample_audio/ — cặp audio + .txt cùng tên (stem)

   Đã kèm sẵn 30 giọng (ví dụ Bá-Vinh.mp3 + Bá-Vinh.txt).

   Nguồn: thư mục audio/ của ViZipVoice trên Hugging Face
     https://huggingface.co/contextboxai/ViZipvoice
     (Apache-2.0 — credit ZipVoice gốc + tác giả ViZipVoice)

   Định dạng audio: .wav, .mp3, .flac, .ogg, .m4a, .opus, .aac

   ID trong app: sample_audio__<tên file>



   Thêm giọng mới: copy vào assets/sample_audio/ rồi bấm «Làm mới danh sách» trên GUI.



Sau khi sửa JSON hoặc thêm file trong sample_audio/, bấm "Làm mới danh sách" trên GUI.



Checkpoint TTS: hynt/ZipVoice-Vietnamese-2500h — Nguyen Thien Hy (hynt)

  https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h

ZipVoice gốc: https://github.com/k2-fsa/ZipVoice

Vocos ONNX (bundled, 100 mel): models/vocoder/mel_spec_24khz.onnx
  Export from ZipVoice-Vietnamese-GUI (Tab Export → Export Vocos ONNX)
  Base weights (attribution): charactr/vocos-mel-24khz

Giấy phép weights: models/THIRD_PARTY_LICENSES.md



Chuẩn hóa text (GUI): pipeline tuần tự — VieNeu, Cấu trúc TTS, sea-g2p.

Nguồn GitHub — xem README.md / README_EN.md:

  VieNeu (port)     https://github.com/pnnbao97/VieNeu-TTS

  Cấu trúc TTS      period_linebreak.py — mã trong repo này

  sea-g2p           https://github.com/pnnbao97/sea-g2p



LICENSE: chỉ nghiên cứu / giáo dục / phi lợi nhuận — cấm thương mại.


