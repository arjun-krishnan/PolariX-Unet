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
from scipy.signal import convolve, fftconvolve
from scipy.ndimage import shift
from tqdm import tqdm
from pathlib import Path
import cv2, h5py, datetime
from scipy.ndimage import convolve as conv_2d
from skimage.measure import label

#%%
def estimate_noise(image):
    # This is a simple placeholder implementation, replace it with your own noise estimation method.
    mean_noise = np.mean(image)
    std_noise = np.std(image)
    status = '' if std_noise > 0 else 'warning'
    return mean_noise, std_noise, status


def get_ROI(image, threshold_factor=1e-3, bits=12, disp_choice=0):
    # Convert image to double precision (float64 in Python)
    image = image.astype(np.float64)

    # Define parameters
    status = ''
    beam_i_threshold = 0.122 * 2**bits  # if bits=12, beam_i_threshold=500

    # Gaussian-like 7x7 kernel
    M = np.array([
        [0.0013, 0.0041, 0.0079, 0.0099, 0.0079, 0.0041, 0.0013],
        [0.0041, 0.0124, 0.0241, 0.0301, 0.0241, 0.0124, 0.0041],
        [0.0079, 0.0241, 0.0470, 0.0587, 0.0470, 0.0241, 0.0079],
        [0.0099, 0.0301, 0.0587, 0.0733, 0.0587, 0.0301, 0.0099],
        [0.0079, 0.0241, 0.0470, 0.0587, 0.0470, 0.0241, 0.0079],
        [0.0041, 0.0124, 0.0241, 0.0301, 0.0241, 0.0124, 0.0041],
        [0.0013, 0.0041, 0.0079, 0.0099, 0.0079, 0.0041, 0.0013]
    ])

    # Check noise level in the image
    mean_noise1, _, status_noise1 = estimate_noise(image)
    if status_noise1 == 'warning':
        status += '- Noise estimate (1) failed -'

    # Subtract noise and filter the image
    image = image - mean_noise1
    image_filt = conv_2d(image, M)

    # Recalculate noise after filtering
    mean_noise2, std_noise2, status_noise2 = estimate_noise(image_filt)
    if status_noise2 == 'warning':
        status += '- Noise estimate (2) failed -'

    # Set the threshold for the ROI detection
    threshold = mean_noise2 + threshold_factor * std_noise2
    #print(threshold)
    
    # Find ROI by thresholding
    num_pix_x, num_pix_y = image_filt.shape
    candidate_list = np.argwhere(image_filt >= threshold)
    num_candidates_start = candidate_list.shape[0]
    
    if num_candidates_start == 0:
        status += '- ROI not found -'

    # Create a binary ROI mask
    roi = np.zeros_like(image_filt)
    roi[image_filt >= threshold] = 1

    # Find the starting point for the ROI selection using a simple average filter
    image_filt_start = convolve(image_filt, np.ones((5, 5)) / 25)
    start_roi_x = np.argmax(np.max(image_filt_start, axis=0))
    start_roi_y = np.argmax(np.max(image_filt_start, axis=1))

    # Label connected components and select the region around the start point
    labeled_roi = label(roi)
    selected_label = labeled_roi[start_roi_y, start_roi_x]
    roi = (labeled_roi == selected_label).astype(int)
    
    # plt.imshow(roi)
    
    # Multiply the ROI mask with the original image
    image_roi = roi * image

    # Set negative pixels to zero
    image_roi = np.maximum(image_roi, 0)
    
    return image_roi


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
        plt.figure()
        plt.plot(x/5, y, linewidth=2, label='original')
        #plt.plot(x, gaussian(x, *params))
        plt.plot(x/5, y_sub, '--', label='subtracted')
        plt.plot(x/5, y_add, '--', label='added')
        plt.plot(x/5, y_aug, linewidth=2, label='augmented')
        plt.legend()
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


def aug_convolution(x, y, a, scale, debug=False):
    if scale < 0.02:
        return y
    
    x_k = np.linspace(0, 10, len(x))
    kernel = gamma.pdf(x_k, a, scale=scale)
    y_aug = convolve(y, kernel, mode='full') / sum(kernel)
    
    if debug == True:
        y_ax = np.linspace(0, len(y), len(y)) 
        y_ax_aug = np.linspace(0, len(y_aug), len(y_aug)) 
        plt.figure()
        plt.plot(y_ax/5,kernel)
        plt.figure()
        plt.plot(y_ax/5,y)
        plt.plot(y_ax_aug/5,y_aug)

    return y_aug


def align_image_off(image_on, image_off):
    
    correlation = fftconvolve(image_on, image_off[::-1, ::-1], mode='full')

    # Step 2: Find the location of the peak in the cross-correlation matrix
    y_peak, x_peak = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Step 3: Calculate the shift required to align the images
    # The center of the cross-correlation matrix corresponds to zero shift
    y_shift = y_peak - (image_on.shape[0] - 1)
    x_shift = x_peak - (image_on.shape[1] - 1)
    
    aligned_image_off = shift(image_off, shift=(y_shift, x_shift), mode='constant', cval=0.0)
    aligned_image_off = get_ROI(aligned_image_off, 1e-4)
    
    return aligned_image_off
    
    
def augmentations(image, image_ref, aug_conv=False):
    
    img, img_ref = np.copy(image), np.copy(image_ref)
    
    kernel_size = 5
    
    img = cv2.medianBlur(img, kernel_size)
    img = get_ROI(img, 1e-4)
    
    img_ref = cv2.medianBlur(img_ref, kernel_size)
    img_ref = get_ROI(img_ref, 1e-4)
    
    img = img / np.max(img)
    img_ref = img / np.max(img_ref)
    
    
    img_aug = np.copy(img)
    
    tx = np.random.randint(-80,80)
    ty = np.random.randint(-50,50)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    translated_image = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
    img_ref = align_image_off(translated_image, img_ref)
    
    x = np.array([i for i in range(img.shape[0])])
    w = np.array([i for i in range(img.shape[-1])])
    
    heaviside_window = np.random.choice([True, False])
    if heaviside_window:
        window_width = np.random.randint(30, 100)
        window_center = np.random.randint(400, 600) #(400,600)
        #window_width = np.random.randint(150, 350)
        epsilon = np.random.randint(8, 12)
        window = heaviside(w, window_center, window_width, epsilon=epsilon)
    else:

        #window_center = np.random.randint(300, 500, 2)
        #window_width = np.random.randint(150, 350, 1)
        window_center = np.random.randint(400, 600, 1) #(400,600)
        window_width = np.random.randint(30, 100, 1) #(30, 100)
        window_amp = np.random.uniform(0.8, 1, 1)
        window = gaussian(w, window_amp[0], window_center[0], window_width[0]/2.355) #+ \
                # gaussian(w, window_amp[1], window_center[1], window_width[1])
    
    conv_a = 1.2
    
    scale = window * 1 * (1 + random_waveform(len(window), 10, t_max=40))
    
    aug_sd = np.random.uniform(2.5, 3.0, len(scale))
    aug_x0 = np.random.uniform(0.25, 0.3, len(scale))

    for i in range(len(w)):
            
        y = translated_image.T[i]

        if aug_conv:
            y_aug = aug_convolution(x, y, conv_a, scale[i])
            idx = int(log_fn(scale[i], *params))
            img_aug.T[i] = y_aug[idx:idx + len(y)] * (1 + (random_waveform(len(y), 10, t_max=10) * scale[i]))
        #    if i == 450:
        #        y_aug = aug_convolution(x, y, conv_a, scale[i], debug=True)
        else:
            y_aug = aug(x, y, scale[i], aug_sd[i], aug_x0[i]*scale[i], debug=False) 
            
         #   if i == 450:
         #       y_aug = aug(x, y, scale[i], aug_sd[i], aug_x0[i]*scale[i], debug=True)
                
            img_aug.T[i] = y_aug * (1 + (random_waveform(len(y), 10, t_max=10) * scale[i]))
    
    #img_aug = cv2.medianBlur(img_aug, kernel_size)
    img_aug = get_ROI(img_aug, 1e-4)
    
    M, N = (np.array(img.shape) // 5)

    img = cv2.resize(translated_image, (N, M))
    img_aug = cv2.resize(img_aug, (N, M))
    img_ref = cv2.resize(img_ref, (N, M))
    
    return img, img_ref, img_aug


def make_train_data(filepath):
    
    train_X = []
    train_Y = []
    train_ref = []
    
    with h5py.File(filepath,'r') as fi:
        imarr = np.array(fi['/zraw/FLASH.DIAG/CAMERA/OTR9FL2XTDS/dGroup/value']).astype(np.float32) #[:,600:1300,600:2000].astype(np.float32)
        for im in tqdm(imarr[:500], desc='Images'):
            for i in range(3):
                idx = np.random.randint(len(imarr))
                im_ref = imarr[idx]
                aug_conv = np.random.choice([True, False])
                imgy, img_ref, imgx = augmentations(im, im_ref, aug_conv)
                train_X.append(imgx)
                train_Y.append(imgy)
                train_ref.append(img_ref)

    print(f"Size of the training set : {len(train_X)}")
    return train_X, train_ref, train_Y

def make_train_data_npy(filepath):
    sase_off_dir = Path(filepath)
    sase_off_files = [file for file in sase_off_dir.glob("*.npy")]
    print(sase_off_files)
    train_X = []
    train_Y = []

    for sase_off in tqdm(sase_off_files[:], desc='Files'):
        images_sase_off = np.load(sase_off).astype(np.float32)
        for img_off in tqdm(images_sase_off, desc='Images'):
            for i in range(4):
                aug_conv = np.random.choice([True, False])
                imgy, imgx = augmentations(img_off, aug_conv)
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
    
    form = "h5" # "h5" for polarix data from desy cloud "npy" for data from sciebo
    
    if form == "h5":
        
        filepath = "run_50631_data.h5"
        train_X, train_ref, train_Y = make_train_data(filepath)
    
        train_data = pd.DataFrame({'train_X' : train_X , 'train_ref' : train_ref, 'train_Y' : train_Y})
        
        current_datetime = datetime.datetime.now()
        datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        
        train_data.to_pickle(f'data/train_data_50631-{datetime_string}.pkl')
        
    if form == "npy":
        
        filepath = "data/sase-off"
        train_X, train_Y = make_train_data_npy(filepath)

        train_data = pd.DataFrame({'train_X' : train_X , 'train_Y' : train_Y})
        train_data.to_pickle('data/train_data_2023_11-08.pkl')
