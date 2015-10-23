#-*-coding:utf-8-*-
from __future__ import division

__author__ = "Frank Jiang"
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as st
import time

time.clock()
p = 0.1
signal_power, noise_power = 1, 0.0
noise_power_times = 50
test_times = 100
length = 1000

SNR_dB = np.zeros((noise_power_times, 1)) 
BER_mean_MAP = np.zeros((noise_power_times, 1))
BER_mean_ML = np.zeros((noise_power_times, 1)) 
BER_all_MAP = np.zeros((test_times, 1)) 
BER_all_ML = np.zeros((test_times, 1))

for i in np.arange(noise_power_times):
    noise_power, noise_mean = noise_power+(i+1)/10, 0
    noise_var = noise_power*0.5
    for j in np.arange(test_times):
        signal = st.binom.rvs(n = 1, p = p, size = length)
        signal = (signal-0.5)*2
        noise = st.norm.rvs(loc = noise_mean, scale = noise_var, size = length)
        mixed_signal = signal + noise
        judged_signal_MAP = np.zeros_like(signal)
        judged_signal_ML = np.zeros_like(signal)
        for k in np.arange(length):
            judged_signal_MAP[k] = 1 if (mixed_signal[k] >= 0.25*noise_power*math.log((1-p)/p)) else -1
            judged_signal_ML[k] = 1 if (mixed_signal[k] >= 0) else -1
        difference_MAP = signal - judged_signal_MAP
        difference_ML = signal - judged_signal_ML
        bit_error_MAP, bit_error_ML = np.linalg.norm(difference_MAP, 0), np.linalg.norm(difference_ML, 0)
        #print bit_error_MAP, bit_error_ML
        BER_all_MAP[j] = bit_error_MAP/length
        BER_all_ML[j] = bit_error_ML/length
    #print BER_all_MAP
    #print BER_all_ML
    BER_mean_MAP[i], BER_mean_ML[i] = np.mean(BER_all_MAP), np.mean(BER_all_ML)
    #print BER_mean_MAP[i], BER_mean_ML[i]
    SNR_dB[i] = 10*math.log(signal_power/noise_power, 10)

print "Elapsed time is %s seconds."%time.clock()
#print BER_mean_MAP
#print BER_mean_ML
plt.plot(SNR_dB, BER_mean_MAP)
plt.plot(SNR_dB, BER_mean_MAP, 'bo')
plt.plot(SNR_dB, BER_mean_ML)
plt.plot(SNR_dB, BER_mean_ML, 'r*')
plt.show()


