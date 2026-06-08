Drop-in ONNX bundle (copy all 5 files → run_gui.bat, no manual quant selection):

  text_encoder_int4.onnx   (or _int8)
  fm_decoder_int4.onnx
  model.json
  tokens.txt
  quantization.json        {"mode": "int4", ...}  — optional; auto-detects int4>int8 if omitted

Override: set ZIPVOICE_ONNX_QUANT=int8|int4 before launch.
