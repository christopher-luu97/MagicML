
from faster_whisper import WhisperModel
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
audio_file = os.path.join(os.getcwd(),"data","delta_orig.wav")

model_size = "medium.en"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe(audio_file, beam_size=5, vad_filter=True) # Use Silero Vad

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))