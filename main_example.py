from mainFunction import mainFunction
import pickle
import matplotlib.pyplot as plt
from patient_class import Patient
import os

mainFunction('D:/hypertrophy/patients/', 'D:/hypertrophy/pickle_from_VS_code/')

data_dir = os.listdir('D:/hypertrophy/pickle_from_VS_code/')
pl = []

for i in range(len(data_dir)):
    pickle_in = open('D:/hypertrophy/pickle_from_VS_code/' + data_dir[i],'rb')
    pl.append(pickle.load(pickle_in))
    pickle_in.close
print('Done reading in')