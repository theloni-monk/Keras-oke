from  pydub import AudioSegment, playback
#pydub.AudioSegment.converter = "C:\\Users\\TheoA\\Downloads\\ffmpeg-20200610-9dfb19b-win64-static\\bin"
import numpy as np

def audio_info(audio, name):
    print('\n-------------------------')
    print('INFO FOR:', name)
    print('len ms', len(audio))
    print('frate', audio.frame_rate)
    print('channels', audio.channels)
    print('fwidth', audio.frame_width)
    print('fcount', int(audio.frame_count()))
    print('swidth', audio.sample_width)
    print('scount', len(audio.get_array_of_samples()))  

def invert(audio):
    #samples = audio.get_array_of_samples() # array of samples between -2^15 and +2^15
    inverse = []
    #exit()
    samples = audio.get_array_of_samples()
    for i in range(len(samples)):
        sample = samples[i]
        sample_val = sample
        inv_sample = -sample_val # when combined will result in silence

        inverse.append(inv_sample.to_bytes(audio.sample_width, byteorder = 'big', signed = True))
    print('inv_scount', len(inverse))
    return AudioSegment(
        data = b''.join(inverse),
        channels = audio.channels,
        sample_width = audio.sample_width,
        frame_rate = audio.frame_rate
    )

# files
                                                                         
combo = AudioSegment.from_mp3("So_Into_You.mp3")
inst_crop = AudioSegment.from_mp3("So_Into_You_inst.mp3")[:len(combo)]
audio_info(combo,'combo')
audio_info(inst_crop, 'inst_crop')

inv = invert(inst_crop)
audio_info(inv, 'inversion')

vocal = combo.overlay(inv) # loop to compensate for diff lengths 
audio_info(vocal,'vocal')

playback.play(vocal)