# class to store information of patients for pickling
class Patient:
    study_id = None
    gender = None
    fov = None # field of view: 123.456x123.456 mm
    pathology = None
    dcm_images = []
    pixel_spacing = None
    contours = [] # from contours.con file 
    slicethickness = None
    num_slices = None
    height = None
    weight = None

