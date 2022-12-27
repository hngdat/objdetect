import cv2 as cv
import numpy as np

def detect_vehicles(self, img):
    
    # Detect Objects
    vehicleBoxs = []
    