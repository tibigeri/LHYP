class ModifyImage:
    def crop(self,patient):
        im = patient.dcm_images
        con = patient.contours
        
        # Lists to store min/max x and y values of each contour
        xminl=[]
        xmaxl=[]
        yminl=[]
        ymaxl=[]
        # Loop through the contours and add the values to the lists 
        for i in con:
            if i is None:
                continue
            xmin=300
            xmax=0
            ymin=300
            ymax=0
            for part in i.values():            
                for xy in part:
                    if xy[0]>xmax: xmax=xy[0]
                    if xy[0]<xmin: xmin=xy[0]
                    if xy[1]>ymax: ymax=xy[1]
                    if xy[1]<ymin: ymin=xy[1]
            xminl.append(xmin)
            xmaxl.append(xmax)
            yminl.append(ymin)
            ymaxl.append(ymax)
            
        # Find the middle of the heart on the image by averaging the given values    
        mx = ((sum(xmaxl)/len(xmaxl))+(sum(xminl)/len(xminl)))/2
        my = ((sum(ymaxl)/len(ymaxl))+(sum(yminl)/len(yminl)))/2
        
        # Crop the image, 120x120 
        ret=[]
        for i in range(len(im)):
            ret.append(im[i][int(my)-60:int(my)+60, int(mx)-60:int(mx)+60])

        return ret
