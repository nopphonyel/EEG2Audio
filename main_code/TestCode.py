from main_code.lib.audioImgConversion import audioImgConversion

convertor = audioImgConversion()
audio_path = 'audio_sample/Sentient.wav'
convertor.loadSignal(audio_path)
convertor.setAudioRange()
convertor.genSpectrogram('./sentient_img.png')
