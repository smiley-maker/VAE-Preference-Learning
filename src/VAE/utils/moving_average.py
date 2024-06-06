from src.VAE.utils.imports import *

def moving_average(arr):
    moving_averages = []

    for i in range(len(arr)):
        avr = sum(arr[:i+1])/(i+1)
        moving_averages.append(avr)

    return moving_averages
