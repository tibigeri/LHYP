import pickle 
def pickling(p,folder_out):
    pickle_file = folder_out + p.study_id
    pickle_out = open(pickle_file + '.pickle', 'wb')
    pickle.dump(p, pickle_out)
    pickle_out.close()
    