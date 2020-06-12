from main_code.lib.audioImgConversion import audioImgConversion

convertor = audioImgConversion()
audio_path = 'audio_sample/Sentient_11K.wav'
convertor.loadSignal(audio_path)
convertor.setAudioRange((1, 2))
convertor.setSampleRate(11000)
convertor.genSpectrogram('./sentient_img.png')
