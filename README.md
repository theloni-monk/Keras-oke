# Keras-oke

## Exploring a pipeline for generating karaoke videos via Automatic Speech Recongnition inference

----------

### Dependencies:

* Spleeter

```bash
pip install spleeter 
```
for aeneas pipeline:
* Aeneas
```bash
sudo apt-get install libespeak-dev
pip install numpy
pip install aeneas
```


for whisper pipeline(deprecated): OpenAI whisper

```bash
pip install git+https://github.com/openai/whisper.git 
```

for correct whisper dependency versions, make sure you have CUDA >= 11.1 and run

```bash
pip install torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
