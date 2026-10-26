import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
DEVICE = "cpu"

PROMPT="Fast-paced J-Rock instrumental, energetic anime opening style, driving distorted electric guitar riffs, punchy fast drum beats, prominent melodic bassline, heroic and triumphant mood, high energy alternative rock."

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
model = model.to(DEVICE)
model.eval()

print("Preparando prompt...")

inputs = processor(
    text=[PROMPT],
    padding=True,
    return_tensors="pt"
).to(DEVICE)

print("Generando música...")

with torch.inference_mode():
    audio = model.generate(
        **inputs, do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=3000 # 5 seg (200) -> 15 seg (750)
    )
sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0,0].cpu().float().numpy()

write("musica2.wav", sample_rate, waveform)
print("Música generada :)")
