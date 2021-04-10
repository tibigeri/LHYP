import os
import pickle
from get_patient import get_patient
from pickling import pickling

def mainFunction(folder_in, folder_out):
    for p in os.listdir(folder_in):
        if os.path.exists(os.path.join(folder_out, p + '.pickle')):
            continue
        to_dump = None
        try:
            to_dump = get_patient(folder_in, p)
        except IndexError:
            print("Error: {}".format(p))
        if to_dump != None:
            pickling(to_dump,folder_out)
            print(to_dump.study_id + ' is pickled')
    print('All possible files have been pickled') 
