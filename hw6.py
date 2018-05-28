#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:21:06 2018

@author: abhinavashriraam
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math 


# Converts images to greyscale
# FROM LECTURE NOTES
def convert_to_greyscale(img):
    img = img/255.0
    def average(pixel):
        return (pixel[0]*0.21 + pixel[1]*0.72 + pixel[2]*0.07)
    greyImg = img[:,:,0].copy()
    for rownum in range(img.shape[0]):
        for colnum in range(img.shape[1]):
            greyImg[rownum][colnum] = average(img[rownum][colnum])
    return greyImg


# @function: draws a pink heart shaped frame around the center of an image
# @param im: normalized np.array input image
# @return im2: normalized, np.array version of input image with heart shaped frame
def heart(im):
    im2 = im.copy()
    y0 = im.shape[0]/4.0 # Sets the center of the first half-circle
    x0 = im.shape[1]/4.0 
    r = im.shape[1]/4.0 # Radius of the first half-circles
    pink = [1.0, 0.753, 0.796] # Color code of pink (RGB)
    for y in range(0,im.shape[0]/4):
        for x in range(0,im.shape[1]/2):
            if math.sqrt((y-y0)**2 + (x-x0)**2) > r: # Defines the area outside the half circle
                im2[y][x] = [1.0, 0.753, 0.796]
    x0 = 3*x0 # shifts the center to be the center of the second half circle
    for y in range(0, im.shape[0]/4):
        for x in range(im.shape[1]/2, im.shape[1]):
            if math.sqrt((y-y0)**2 + (x-x0)**2) > r: # defines the region outside the half circle
                im2[y][x] = [1.0, 0.753, 0.796]
    y0 = im.shape[0] - y0 # Changes the y system to the traditional co-ordinate axes
    x0 = 0 # Points the line passes through
    y1 = 0
    x1 = im.shape[1]/2.0
    m = (y1-y0)/(x1 - x0) # Slope
    for y in range(im.shape[0]/4, im.shape[0]):
        for x in range(0, im.shape[1]/2):
            y3 = im.shape[0] - y 
            if y3 - y0 < m*(x): # Region outside the line
                im2[y][x] = [1.0, 0.753, 0.796] # Set to pink
    x0 = im.shape[1] # New set of points
    m = (y1 - y0)/(x1 - x0) # Slope
    for y in range(im.shape[0]/4, im.shape[0]):
        for x in range(im.shape[1]/2,im.shape[1]):
            y3 = im.shape[0] - y
            if y3 - y0 < m*(x - x0): # Region outside the line
                im2[y][x] = [1.0, 0.753, 0.796]       
    return im2

    
# @function: does noise removal (uniform or Gaussian) on an input image
# @param im: input image
# @param method: style of noise removal - uniform or Gaussian
# @return im2: output image, with blurring
def blurring(im,method):
    im2 = im.copy()
    if method == "uniform":
        k = 5
        for i in range(2, im2.shape[0]-2):
            for j in range(2, im2.shape[1]-2):
                neighborhood = im2[(i - (k-1)/2):(i + (k-1)/2) + 1, (j - (k-1)/2):(j + (k-1)/2) + 1] # Defines the 5x5 sub-array of neighboring pixels
                filter = np.array([[1.0/5**2]*5]*5) # 5x5 matrix of entries 1/25
                neighborhood = neighborhood * filter # element wise multiplication
                im2[i][j] = np.sum(neighborhood) # composition
        return im2
    if method == "Gaussian":
        k = 5
        sigma = 1
        for i in range(2, im2.shape[0]-2):
            for j in range(2, im2.shape[1]-2):
                filter=np.array([[0]*k]*k,dtype='float')
                for x in range(k):
                    for y in range(k):
                        filter[x,y]= (np.exp(-((x-(k-1)*0.5)**2+(y-(k-1)*0.5)**2)/(2.0*sigma**2)))*(1.0/(2*math.pi*(sigma**2))) # Gaussian filter
                if np.sum(filter) < 1:
                    filter = filter/np.sum(filter)
                neighborhood = im2[(i - (k-1)/2):(i + (k-1)/2) + 1, (j - (k-1)/2):(j + (k-1)/2) + 1]
                neighborhood = neighborhood*filter
                im2[i][j] = np.sum(neighborhood)
        return im2
                
        
        
# @function: helper function to perform  compositions on matrices
# @param A,B: matrices of equal dimension
# @return: the composition
def composition(A,B):
    if A.shape != B.shape:
        print "Dimensions are not perfect"
    product = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            product[i][j] = A[i][j]*B[i][j] # Element-wise multiplication
    return np.sum(product) # Returns the sum
  
    
# @function: detects edges, returns a greyscale image of edge detection
# @param im: input image im
# @param method: Choice of "horizontal", "vertical", or "both"
# @return: image with edge detection
def detect_edge(im, method):
    horizontal_filter = np.array([[-1,0,1], [-2,0,2], [-1,0,1]]) # Filters from lecture notes
    vertical_filter = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])  
    edge = im
    m = im.copy()
    rownum = im.shape[0]
    colnum = im.shape[1]
       
    for i in range(0, rownum - 2):
        for j in range(0, colnum - 2):
            
            subarray = m[i: (i + 3), :]
            subarray = subarray[:, j:(j+3)] # code to obtain the 3x3 subarray surrounding the pixel
            s_h = composition(subarray, horizontal_filter) # from lecture notes
            s_v = composition(subarray, vertical_filter)
            
            if method == "horizontal":
                edge[i][j] = (s_h + 4)/8.0 # Normalizes (range -4 to 4)
            elif method == "vertical":
                edge[i][j] = (s_v + 4)/8.0 # Normalizes (range -4 to 4)
            elif method == "both":
                edge[i][j] = (np.sqrt(s_h**2 + s_v**2))/4.0 # Normalizes
    return edge
        

