import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
    # To do
    filter_x = np.array([[1, 0, -1],
                  [ 1, 0, -1],
                  [ 1, 0, -1]])
    
    filter_y = np.array([[1,1,1],
                  [ 0, 0, 0],
                  [ -1, -1, -1]])
    
    return filter_x, filter_y


def filter_image(im, filter):
    # To do    
    dim1,dim2 = im.shape    
    im_f = np.zeros((dim1+2,dim2+2), dtype = np.int)
    for i in range(1,dim1+1):
        for j in range(1,dim2+1):
            im_f[i,j] = im[i-1,j-1]                
    im_filtered = np.zeros((dim1,dim2), dtype = np.int)    
    for i in range(0,dim1):
        for j in range(0,dim2):
            im_filtered[i,j] = (im_f[(i):(i+3),(j):(j+3)]*filter).sum()
    plt.imshow(im_filtered,cmap='winter')
    plt.axis("on")            
    return im_filtered


def get_gradient(im_dx, im_dy):
    # To do
    grad_mag = np.sqrt(im_dx**2+im_dy**2)
    dim1,dim2 = im_dx.shape
    
    grad_angle = np.zeros((dim1,dim2), dtype = np.int)
    for i in range(dim1):
        for j in range(dim2):
            if (im_dy[i,j]!=0):
                grad_angle[i,j] = np.arctan(im_dx[i,j]/im_dy[i,j])*180/np.pi            
            if (im_dx[i,j]==0 and im_dy[i,j]==0):           
                grad_angle[i,j] = 0
            if (im_dx[i,j]!=0 and im_dy[i,j]==0):
                grad_angle[i,j] = 90
            if (grad_angle[i,j]<0):
                grad_angle[i,j] += 180 
    plt.imshow(grad_mag,cmap='winter')
    plt.axis("on")
    plt.imshow(grad_angle,cmap='winter')
    plt.axis("on")    
    return grad_mag, grad_angle

def cell_gradient(cell_magnitude, cell_angle, bin_size):
    histo = [0] * bin_size
    angle_unit = 180 / bin_size
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]
            gradient_angle = cell_angle[k][l]
            mod = (int)(gradient_angle/angle_unit)
            histo[mod] += gradient_strength          
    return histo

def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    bin_size = 6
    grad_mag = abs(grad_mag)
    ori_histo = np.zeros(((int)(grad_angle.shape[0] / cell_size), (int)(grad_angle.shape[1] / cell_size), bin_size))

    for i in range(ori_histo.shape[0]):
        for j in range(ori_histo.shape[1]):
            cell_magnitude = grad_mag[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell_angle = grad_angle[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            #print cell_angle.max()
            ori_histo[i][j] = cell_gradient(cell_magnitude, cell_angle, bin_size)    
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    ori_histo_normalized = np.zeros((ori_histo.shape[0] - 1, ori_histo.shape[1] - 1, block_size*block_size*6), dtype=np.float)
    total_mag = 0
    for i in range(ori_histo.shape[0] - 1):
        for j in range(ori_histo.shape[1] - 1):
            total_mag = 0
            block_vector = []
            block_vector.extend(ori_histo[i][j])
            block_vector.extend(ori_histo[i][j + 1])
            block_vector.extend(ori_histo[i + 1][j])
            block_vector.extend(ori_histo[i + 1][j + 1])            
            for k in range (6):
                total_mag += ori_histo[i,j,k]*ori_histo[i,j,k]
                total_mag += ori_histo[i+1,j,k]*ori_histo[i+1,j,k]
                total_mag += ori_histo[i,j+1,k]*ori_histo[i,j+1,k]
                total_mag += ori_histo[i+1,j+1,k]*ori_histo[i+1,j+1,k]
            ori_histo_normalized[i,j, ] = np.array(block_vector)/np.sqrt(total_mag + 0.001*0.001)
            #print(ori_histo_normalized[i,j])    
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    # To do
    image = np.array(im)
    filter_x, filter_y = get_differential_filter()
    im_filtered_x = filter_image(image, filter_x)
    im_filtered_y = filter_image(image, filter_y)
    grad_mag,grad_angle = get_gradient(im_filtered_x, im_filtered_y)
    ori_histo = build_histogram(grad_mag, grad_angle, 8)
    ori_histo_normalized = get_block_descriptor(ori_histo, 2)
    #print(ori_histo_normalized)
    # visualize to verify
    visualize_hog(im, ori_histo_normalized, 8, 2)
    #plt.imshow(im_filtered_x,cmap='winter', vmin=0, vmax=1)
    #plt.show

    return ori_histo_normalized


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    im = im.astype('float') / 255.0
    num_bins = 6
    max_len = 7 # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))

    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, 180, 180/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[int(cell_size*block_size/2): cell_size*num_cell_w-(cell_size*block_size/2)+1: cell_size], np.r_[int(cell_size*block_size/2): cell_size*num_cell_h-(cell_size*block_size/2)+1: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins)) # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins)) # expand to same dims as histo_normalized
    plt.figure("HOG")
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i], color='white',headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


if __name__=='__main__':
    ori_image = cv2.imread('cameraman.tif', 0)
    im = cv2.GaussianBlur(ori_image,(5,5),1)    
    hog = extract_hog(im)
    


