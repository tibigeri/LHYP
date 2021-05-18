import pickle 
from crop_image import ModifyImage
import os

class saveToPickle:

    def picklePatient(self,p,folder_out):
        pickle_file = folder_out + p.study_id
        #pickle_file = pickle_file.strip('\r')
        pickle_out = open(pickle_file + '.pickle', 'wb')
        pickle.dump(p, pickle_out)
        pickle_out.close()
    
    def pickleCroppedImages(self, folder_in, folder_out):
        p_folder = os.listdir(folder_in)
        mi = ModifyImage()

        for i in range(len(p_folder)):
            pickle_in = open(folder_in + p_folder[i],'rb')
            temp = pickle.load(pickle_in)
            pickle_in.close

            temp.dcm_images = mi.crop(temp)

            
            pickle_file = folder_out + temp.study_id.split()[0] #.split()[0] was needed because some study_ids had '\r' at the end
            pickle_out = open(pickle_file + 'c.pickle', 'wb')
            pickle.dump(temp, pickle_out)
            pickle_out.close()


