# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:13:31 2024

@author: arjun
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import gamma
from scipy.signal import convolve
from tqdm import tqdm
import glob
import cv2
import h5py
import datetime

#%%

def gaussian(x, a, x0, sd):
    return a * np.exp(-0.5 * ((x - x0) / sd)**2)


def aug(x, y, a_sub, sd_add, x0_add, debug=False):
    
    if a_sub < 0.02:
        return y
    
    initial_guess = [max(y), np.argmax(y), len(x)/10]
    
    try:
        params, cov = curve_fit(gaussian, x, y, p0=initial_guess)
    except:
        return y

    a_fit, x0_fit, sd_fit = params

    y_sub = gaussian(x, a=a_sub*a_fit, x0=x0_fit, sd=sd_fit)

    a_add = a_sub*a_fit/sd_add
    x0_add = x0_fit + x0_add * len(x)
    y_add = gaussian(x, a_add, x0_add, sd_add*sd_fit)
    y_aug = y - y_sub + y_add
    y_aug = np.clip(y_aug, a_min=0.0, a_max=255.0)
    
    if debug:
        print(params)
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, gaussian(x, *params))
        plt.plot(x, y_sub)
        plt.plot(x, y_add)
        plt.plot(x, y_aug)
    return y_aug

def heaviside(x, center, width, epsilon=1.0):
    e1 = center - width/2
    e2 = center + width/2

    half1 = 0.5 + 0.5 * (2/np.pi) * np.arctan((x-e1)/epsilon)
    half2 = 1 - (0.5 + 0.5 * (2/np.pi) * np.arctan((x-e2)/epsilon))

    window = np.concatenate((half1[:center], half2[center:]))

    return window

def random_waveform(num_samples, num_components, t_max):
    # Generate random frequencies, phases, and amplitudes for each component
    frequencies = np.random.uniform(0.5, 2, num_components)  # Random frequencies between 0.5 and 2 Hz
    phases = np.random.uniform(0, 2*np.pi, num_components)  # Random phases between 0 and 2*pi
    amplitudes = np.random.uniform(0.02, 0.025, num_components)  # Random amplitudes between 0.5 and 2

    # Time array
    time = np.linspace(0, t_max, num_samples)

    # Generate waveform by summing sine and cosine components
    waveform = np.zeros(num_samples)
    for i in range(num_components):
        waveform += amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * time + phases[i])  # Add sine component
        waveform += amplitudes[i] * np.cos(2 * np.pi * frequencies[i] * time + phases[i])  # Add cosine component

    return waveform


def plot_polarix(filename):
    images = np.load(filename).astype('float32')
    N = int(images.shape[0])
    a = 5
    b = int(np.ceil(N / a))
    fig, axs = plt.subplots(a, b)

    for i in range(N):
        row = i // b
        col = i % b
        axs[row, col].imshow(images[i])

def aug_convolution(x, y, a, scale):
    if scale < 0.02:
        return y
    
    x_k = np.linspace(0, 10, len(x))
    kernel = gamma.pdf(x_k, a, scale=scale)
    y_aug = convolve(y, kernel, mode='full') / sum(kernel)

    return y_aug

def augmentations(image, aug_conv=False):
    img = np.copy(image)
    img = img / np.max(img)
    kernel_size = 5
    img = cv2.medianBlur(img, kernel_size)
    img_aug = np.copy(img)

    tx = np.random.randint(-80,80)
    ty = np.random.randint(-50,50)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    translated_image = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))

    x = np.array([i for i in range(img.shape[0])])
    w = np.array([i for i in range(img.shape[-1])])
    
    heaviside_window = np.random.choice([True, False])
    if heaviside_window:
        window_center = np.random.randint(500,750)
        window_width = np.random.randint(30, 100)
        epsilon = np.random.randint(8, 12)
        window = heaviside(w, window_center, window_width, epsilon=epsilon)
    else:
        window_center = np.random.randint(400, 600, 2)
        window_width = np.random.randint(10, 50, 2)
        window_amp = np.random.uniform(0.3, 0.8, 2)
        window = gaussian(w, window_amp[0], window_center[0], window_width[0]) + \
                 gaussian(w, window_amp[1], window_center[1], window_width[1])
    
    conv_a = 1
    scale = window * 1 * (1 + random_waveform(len(window), 10, t_max=40))
    
    aug_sd = np.random.uniform(2.5, 3.0, len(scale))
    aug_x0 = np.random.uniform(0.25, 0.3, len(scale))
    for i in range(len(w)):
        y = translated_image.T[i]

        if aug_conv:
            y_aug = aug_convolution(x, y, conv_a, scale[i])
            idx = int(log_fn(scale[i], *params))
            img_aug.T[i] = y_aug[idx:idx + len(y)] * (1 + (random_waveform(len(y), 10, t_max=10) * scale[i]))
        else:
            y_aug = aug(x, y, scale[i], aug_sd[i], aug_x0[i]*scale[i], debug=False) 
            img_aug.T[i] = y_aug * (1 + (random_waveform(len(y), 10, t_max=10) * scale[i]))
    
    #img_aug = cv2.medianBlur(img_aug, kernel_size)
    
    M, N = (np.array(img.shape) // 5)
   # img = (cv2.resize(translated_image, (N, M)) * 255 / np.max(img)) .astype(np.uint8)
   # img_aug = (cv2.resize(img_aug, (N, M)) * 255 / np.max(img_aug)).astype(np.uint8)
    img = cv2.resize(translated_image, (N, M))
    img_aug = cv2.resize(img_aug, (N, M))
    
    return img, img_aug


def make_train_data(filepath):
    
    train_X = []
    train_Y = []
    
    with h5py.File(filepath,'r') as fi:
        imarr = np.array(fi['/zraw/FLASH.DIAG/CAMERA/OTR9FL2XTDS/dGroup/value'])[:,:,:].astype(np.float32)
        for im in tqdm(imarr[:], desc='Images'):
            for i in range(3):
                aug_conv = np.random.choice([True, False])
                imgy, imgx = augmentations(im, aug_conv)
                train_X.append(imgx)
                train_Y.append(imgy)

    print(f"Size of the training set : {len(train_X)}")
    return train_X, train_Y

#%%

def log_fn(x, a):
    y = a * np.log(x+1)
    return y

s = [0, 0.1, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
ii = [0, 5, 15, 25, 35, 40, 45, 50, 55, 60, 70]
params, cov = curve_fit(log_fn, s, ii, p0=[40])


#%%
if __name__ == "__main__":

    filepath = "run_50631_data.h5"
    train_X, train_Y = make_train_data(filepath)

    train_data = pd.DataFrame({'train_X' : train_X , 'train_Y' : train_Y})
    
    current_datetime = datetime.datetime.now()
    datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    
    train_data.to_pickle(f'data/train_data_50631-{datetime_string}.pkl')
