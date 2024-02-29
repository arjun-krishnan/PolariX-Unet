# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:13:31 2024

@author: arjun
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.stats import gamma
from pathlib import Path
import cv2

def gaussian(x, a, x0, sd):
    return a * np.exp(-0.5 * ((x - x0) / sd)**2)

def aug(x, y, a_sub, sd_add, x0_add, debug=False):
    
    initial_guess = [max(y), np.argmax(y), len(x)/10]
    
    try:
        params, cov = curve_fit(gaussian, x, y, p0=initial_guess)
    except:
        return(y)
    
    a_fit, x0_fit, sd_fit = params
    
    y_sub = gaussian(x, a=a_sub*a_fit, x0=x0_fit, sd=sd_fit)

    a_add = a_sub*a_fit*sd_fit/(sd_add*sd_fit)
    x0_add = x0_fit + x0_add * len(x) 
    y_add = gaussian(x, a_add, x0_add, sd_add*sd_fit)
    y_aug = y - y_sub + y_add
    y_aug = np.clip(y_aug, a_min = 0.0, a_max = 255.0)
    
    if debug:
        print(params)
        plt.figure()
        plt.plot(x,y)
        plt.plot(x,gaussian(x,*params))
        plt.plot(x,y_sub)
        plt.plot(x,y_add)
        plt.plot(x, y_aug)
    return(y_aug)

def heaviside(x, center, width, epsilon=1):
    e1 = center - width/2
    e2 = center + width/2
    
    half1 = 0.5 + 0.5 * (2/np.pi) * np.arctan((x-e1)/epsilon)
    half2 = 1 - (0.5 + 0.5 * (2/np.pi) * np.arctan((x-e2)/epsilon))
    
    window = np.concatenate((half1[:center], half2[center:]))
    
    return window

def log_fn(x, a):
    y = a * np.log(x+1)
    return y

def aug_convolute(x,y, a, scale):
    x_k = np.linspace(0,10,len(x))
    kernel = gamma.pdf(x_k, a, scale=scale)
    y_aug = convolve(y, kernel, mode='full') / sum(kernel)
    
    return y_aug

def random_waveform(num_samples, num_components):
    # Generate random frequencies, phases, and amplitudes for each component
    frequencies = np.random.uniform(0.5, 2, num_components)  # Random frequencies between 0.5 and 2 Hz
    phases = np.random.uniform(0, 2*np.pi, num_components)  # Random phases between 0 and 2*pi
    amplitudes = np.random.uniform(0.02,0.025, num_components)  # Random amplitudes between 0.5 and 2

    # Time array
    time = np.linspace(0, 40, num_samples)

    # Generate waveform by summing sine and cosine components
    waveform = np.zeros(num_samples)
    for i in range(num_components):
        waveform += amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * time + phases[i])  # Add sine component
        waveform += amplitudes[i] * np.cos(2 * np.pi * frequencies[i] * time + phases[i])  # Add cosine component
        
    return waveform    

data_dir = Path("data")

images_sase_off = np.load(data_dir / "lhpulses_zero-sase_off-polarix-2023-11-08T032809.npy").astype('float32')
images_sase_on  = np.load(data_dir / "lhpulses_zero-sase_on-polarix-2023-11-08T032511.npy").astype('float32')

img = np.copy(images_sase_off[6])

kernel_size = 5
img = cv2.medianBlur(img, kernel_size)

img_aug = np.copy(img)
x = np.array([i for i in range(img.shape[1])])
w = np.array([i for i in range(img.shape[-1])])
window = heaviside(w, 450,100, epsilon=10.5)

s = [0,0.1,0.3,0.5,1,1.5,2,2.5,3,3.5,4]
ii = [0,5,15,25,35,40,45,50,55,60,70]
params, cov = curve_fit(log_fn, s, ii, p0=[40])

a = 1
scale = window * 1 * (1 + random_waveform(len(window), 10))
                      
for i in range(len(w)):
    #scale_aug = scale[i] * (1 + random_waveform(len(window), 10))[i]
    y = img.T[i]
    #y_aug = aug(x,y, a_sub=0.8*window[i], sd_add = 5, x0_add=0.2)
    y_aug = aug_convolute(x, y, a, scale[i])
    idx = int(log_fn(scale[i], *params))     # This 42 is from fitting the scale and shift values
    noise = np.random.normal(1, scale[i]/10, len(y))
    img_aug.T[i] = y_aug[idx:idx+len(y)] #* (1 + (random_waveform(len(y), 10) * scale[i]))#* noise 

M, N = (np.array(img.shape) / 5)
img = cv2.resize(img,(int(N),int(M)))
img_aug = cv2.resize(img_aug,(int(N),int(M)))
    
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(img_aug)
plt.show()