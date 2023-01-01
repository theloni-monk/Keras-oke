import sys
from dataclasses import dataclass
import re
from typing import Union, TextIO
import numpy as np

import chars2vec
import whisper


@dataclass 
class transcript_element:
    start: float
    end: float
    phrase: str

@dataclass 
class transcription:
    script: list[transcript_element]

#WRITEME: wrap spleeter api to extract vocals

def generate_transcript(whisper_model, c2v_model, audio_target: Union[str, np.ndarray], transcript_prior: str):
    whisper_output = whisper.transcribe(whisper_model, audio_target)
    
    whisper_transcript = transcription([transcript_element(seg['start'], seg['end'], seg['text']) for seg in whisper_output['segments']])

    lyrics = transcript_prior.strip("(),")
    lyrics = re.split(' |\n|, ', lyrics)
    lyrics_embedding = c2v_model.vectorize_words(lyrics) # sequence of vectors of dimension 50

    # match whisper transcription to known prior (the fetched lyrics)
    matched_transcript = transcription([])
    lyric_idx = 0
    target_locs = []
    for script_segment in whisper_transcript.script:
        seg_words = script_segment.phrase.strip()
        seg_words = ''.join([i for i in seg_words if i.isalpha() or i == ' ']).split(' ')
        num_words = len(seg_words)

        seg_embedding = c2v_model.vectorize_words(seg_words) # sequence of vectors of dimension 50
        # compute embedding distance between sequences in the lyric and the estimate sequence
        correlation = []
        for i in range(lyric_idx, len(lyrics) - num_words): #TODO: shorten this search range
            embedding_distance = np.linalg.norm(lyrics_embedding[i:i+num_words] - seg_embedding)
            correlation.append(embedding_distance)
        correlation = [max(correlation) - i for i in correlation]
        correlation = np.array(correlation)

        # look for when the segment phrase starts
        try:
            amax = np.argmax(correlation[:num_words+3]) # 3 is a fudge factor for how off the words could be
            target_location = lyric_idx + amax 
        except ValueError:
            target_location = lyric_idx
        target_locs.append(target_location)

        lyric_idx+=num_words

    #FIXME: introducing timing errors
    for i in range(0,len(target_locs)-1):
        script_segment = whisper_transcript.script[i]
        prev_target_loc = target_locs[i]
        target_loc = target_locs[i+1]
        target_lyrics = " ".join(lyrics[prev_target_loc:target_loc])
        matched_transcript.script.append(transcript_element(script_segment.start, script_segment.end, target_lyrics))
    final_seg =  whisper_transcript.script[-1]
    matched_transcript.script.append(transcript_element(final_seg.start, final_seg.end, " ".join(lyrics[target_locs[-1]:len(lyrics)])))

    # filter out empty phrases
    i = 0
    while i < len(matched_transcript.script):
        if matched_transcript.script[i].phrase == [] or matched_transcript.script[i].phrase == ['']:
            temp = matched_transcript.pop(i)
            matched_transcript[i-1][1] = temp[1]
        else: 
            i+=1
    return matched_transcript

def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

# write transcript to closed caption file format for web
def write_vtt(transcript: transcription, file: TextIO):
    print("WEBVTT\n", file=file)
    for segment in transcript.script:
        print(
            f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
            f"{ segment.phrase.replace('-->', '->')}\n",
            file=file,
            flush=True,
        )

# write transcript to closed caption file format
def write_srt(transcript: transcription, file: TextIO):
    for i, segment in enumerate(transcript.script, start=1):
        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
            f"{segment.phrase.strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


instructions = """Generates .vtt closed captioning file given an audio file and its lyrics
Syntax: kerasoke audio_path.mp3 lyrics_path.txt out_path.vtt"""

if __name__ == "__main__":
    whisper_model  = whisper.load_model("base.en")
    c2v_model = chars2vec.load_model('eng_50')

    if sys.argv[1] == "-h":
        print(instructions)
        exit(0)

    audio_path = sys.argv[1]
    prior_path = sys.argv[2]

    print("Processing", audio_path, "with prior", prior_path)

    out_path = sys.argv[3]
    with open(prior_path,"r") as lyric_file, open(out_path, "w") as outfile:
        lyrics = lyric_file.read()
        captions = generate_transcript(whisper_model, c2v_model, audio_path, lyrics)
        
        print("writing captions to", out_path)
        write_vtt(captions, outfile)