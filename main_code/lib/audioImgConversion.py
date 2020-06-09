from scipy.io.wavfile import read as readSound
from scipy.io.wavfile import write as writeSound
from PIL import Image
import math
import numpy as np
import time


def amplifyMagnitudeByLog(d):
    return 188.301 * math.log10(d + 1)


def weakenAmplifiedMagnitude(d):
    return math.pow(10, d / 188.301) - 1


class audioImgConversion:
    def __init__(self):
        self.__rate = 32000  # Default output recovered audio rate
        self.__FFT_LENGTH = 1024
        self.__WINDOW_LENGTH = 512
        self.__WINDOW_STEP = int(self.__WINDOW_LENGTH / 2)
        self.__phaseMin = -math.pi
        self.__phaseMax = math.pi
        self.__phaseRange = self.__phaseMax - self.__phaseMin
        self.__magnitudeMin = float(0)
        self.__magnitudeMax = float(4000000)
        self.__magnitudeRange = self.__magnitudeMax - self.__magnitudeMin
        self.__signal = None
        self.__signalOrig = None

    def __generateLinearScale(self, magnitudePixels, phasePixels):
        height = magnitudePixels.shape[0]
        width = magnitudePixels.shape[1]
        phaseRange = self.__phaseMax - self.__phaseMin
        rgbArray = np.zeros((height, width, 3), 'uint8')

        for w in range(width):
            for h in range(height):
                magnitudePixels[h, w] = (magnitudePixels[h, w] - self.__magnitudeMin) / self.__magnitudeRange * 255 * 2
                magnitudePixels[h, w] = amplifyMagnitudeByLog(magnitudePixels[h, w])
                phasePixels[h, w] = (phasePixels[h, w] - self.__phaseMin) / phaseRange * 255
                red = 255 if magnitudePixels[h, w] > 255 else magnitudePixels[h, w]
                green = (magnitudePixels[h, w] - 255) if magnitudePixels[h, w] > 255 else 0
                blue = phasePixels[h, w]
                rgbArray[h, w, 0] = int(red)
                rgbArray[h, w, 1] = int(green)
                rgbArray[h, w, 2] = int(blue)
        return rgbArray

    def __recoverLinearScale(self, rgbArray):
        width = rgbArray.shape[1]
        height = rgbArray.shape[0]
        magnitudeVals = rgbArray[:, :, 0].astype(float) + rgbArray[:, :, 1].astype(float)
        phaseVals = rgbArray[:, :, 2].astype(float)
        for w in range(width):
            for h in range(height):
                phaseVals[h, w] = (phaseVals[h, w] / 255 * self.__phaseRange) + self.__phaseMin
                magnitudeVals[h, w] = weakenAmplifiedMagnitude(magnitudeVals[h, w])
                magnitudeVals[h, w] = (magnitudeVals[h, w] / (255 * 2) * self.__magnitudeRange) + self.__magnitudeMin
        return magnitudeVals, phaseVals

    def setAudioRange(self, aud_range):
        if aud_range is None:
            self.__signal = self.__signalOrig[:, 0]
        else:
            if type(aud_range) is tuple:
                self.__signal = self.__signalOrig[:, 0][aud_range[0] * self.__rate: aud_range[1] * self.__rate]
            else:
                print("<X> : Parameter expected to be None or Tuple")

    def loadSignal(self, path):
        self.__rate, audData = readSound(path)
        self.__signalOrig = audData
        self.__signal = self.__signalOrig[:, 0]

    def setRecoverRate(self, newRate):
        self.__rate = newRate

    def genSpectrogram(self, exportPath):
        if self.__signal is None:
            print("<X> : Signal not yet loaded")
            return
        start_time = time.time()
        buffer = np.zeros(int(self.__signal.size + self.__WINDOW_STEP - (self.__signal.size % self.__WINDOW_STEP)))
        buffer[0:len(self.__signal)] = self.__signal
        height = int(self.__FFT_LENGTH / 2 + 1)
        width = int(len(buffer) / self.__WINDOW_STEP - 1)
        magnitudePixels = np.zeros((height, width))
        phasePixels = np.zeros((height, width))

        for w in range(width):
            buff = np.zeros(self.__FFT_LENGTH)
            stepBuff = buffer[w * self.__WINDOW_STEP:w * self.__WINDOW_STEP + self.__WINDOW_LENGTH]
            # apply hanning window
            stepBuff = stepBuff * np.hanning(self.__WINDOW_LENGTH)
            buff[0:len(stepBuff)] = stepBuff
            # buff now contains windowed signal with step length and padded with zeroes to the end
            fft = np.fft.rfft(buff)
            for h in range(len(fft)):
                magnitude = math.sqrt(fft[h].real ** 2 + fft[h].imag ** 2)
                phase = math.atan2(fft[h].imag, fft[h].real)
                magnitudePixels[height - h - 1, w] = magnitude
                phasePixels[height - h - 1, w] = phase
        rgbArray = self.__generateLinearScale(magnitudePixels, phasePixels)
        elapsed_time = time.time() - start_time
        print('%.2f' % elapsed_time, 's', sep='')
        img = Image.fromarray(rgbArray, 'RGB')
        img.save(exportPath, "PNG")

    def recoverAudio(self, imgPath, exportPath):
        img = Image.open(imgPath)
        data = np.array(img, dtype='uint8')
        width = data.shape[1]
        height = data.shape[0]

        magnitudeVals, phaseVals = self.__recoverLinearScale(data)

        recovered = np.zeros(self.__WINDOW_LENGTH * width // 2 + self.__WINDOW_STEP, dtype=np.int16)
        for w in range(width):
            toInverse = np.zeros(height, dtype=np.complex_)
            for h in range(height):
                magnitude = magnitudeVals[height - h - 1, w]
                phase = phaseVals[height - h - 1, w]
                toInverse[h] = magnitude * math.cos(phase) + (1j * magnitude * math.sin(phase))
            signal = np.fft.irfft(toInverse)
            recovered[w * self.__WINDOW_STEP:w * self.__WINDOW_STEP + self.__WINDOW_LENGTH] += signal[
                                                                                               :self.__WINDOW_LENGTH].astype(
                np.int16)
        writeSound(exportPath, self.__rate, recovered)
