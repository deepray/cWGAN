
import numpy as np

rand = np.random.rand

class sup_shape:

    def __init__(self,
                 shape_type   = 'ellipse',
                 n_shapes     = 1):
        self.shape_type = shape_type
        self.n_shapes   = n_shapes
        self.max_val    = 1.0  # DO NOT CHANGE!!
        self.set_geomtery()

    def set_geomtery(self):
        if self.shape_type == 'ellipse':
            self.set_ellipse_geom()
        elif self.shape_type == 'rectangle':  
            self.set_rectangle_geom()
        
    def set_ellipse_geom(self):
        self.E = np.hstack((2*(np.random.rand(self.n_shapes,2)-0.5), 
                            2*(np.random.rand(self.n_shapes,2)+1.0), 
                            90*(np.random.rand(self.n_shapes,1)-0.5),
                            0.2*np.random.rand(self.n_shapes,1)+0.7 ))


    def set_rectangle_geom(self):
        self.E = np.hstack((2*(np.random.rand(self.n_shapes,2)-0.5), 
                            2*(np.random.rand(self.n_shapes,2)+1.0), 
                            90*np.random.rand(self.n_shapes,1),
                            0.2*np.random.rand(self.n_shapes,1)+0.7)) 

    def discrete_shape(self,xcoord,ycoord):
        if self.shape_type == 'ellipse':
            image = self.get_disc_ellipse(xcoord,ycoord)
        elif self.shape_type == 'rectangle':  
            image = self.get_disc_rectangle(xcoord,ycoord) 

        return self.scale(image)

    def get_disc_ellipse(self,xcoord,ycoord):    
        image     = np.zeros(xcoord.shape).T.reshape(1,-1)

        for k in range(self.n_shapes):
            Vx0         = np.vstack(  (xcoord.T.reshape(1,-1) - self.E[k,0] , ycoord.T.reshape(1,-1) - self.E[k,1]) )
            D           = np.matrix([[1/self.E[k,2],0],[0,1/self.E[k,3]]])
            phi         = self.E[k,4]*np.pi/180
            Q           = np.matrix( [[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]] )
            f           = self.E[k,5]
            equation1   = np.sum(np.square(D*Q*Vx0),axis=0)
            i           = np.argwhere(equation1<=1.0)[:,1]
            image[0,i]  = image[0,i]+f;

        image = image.reshape(xcoord.shape[1],xcoord.shape[0]).T

        return image


    def get_disc_rectangle(self,xcoord,ycoord):    
        image     = np.zeros(xcoord.shape)

        for k in range(self.n_shapes):
            x0      = self.E[k,0]
            y0      = self.E[k,1]
            a       = self.E[k,2]
            b       = self.E[k,3]
            phi     = self.E[k,4]*np.pi/180 
            phi2    = self.E[k,4]*np.pi/180 + np.pi/2.0
            f       = self.E[k,5]
            line1   = ycoord*np.cos(phi) - xcoord*np.sin(phi) - b -y0*np.cos(phi) - x0*np.sin(phi)
            line2   = ycoord*np.cos(phi) - xcoord*np.sin(phi) + b -y0*np.cos(phi) - x0*np.sin(phi)
            line3   = ycoord*np.cos(phi2) - xcoord*np.sin(phi2) - a -y0*np.cos(phi2) - x0*np.sin(phi2)
            line4   = ycoord*np.cos(phi2) - xcoord*np.sin(phi2) + a -y0*np.cos(phi2) - x0*np.sin(phi2)
            i1      = line1<=0.0
            i2      = line2>=0.0
            i3      = line3<=0.0
            i4      = line4>=0.0
            image  += f*i1*i2*i3*i4;

        return image    


    def scale(self,image):    
        in_max = np.max(image)


        if in_max > self.max_val:    
            scaled_image = image/(in_max)
        else:
            scaled_image = 1.0*image

        return scaled_image 

    





            
