import sys
import cv2
import numpy as np
import scipy as scp
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

def create_binary_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

    # Size kernel to fit image!
    kernel = np.ones((3,3),np.uint8)
    clean_binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # HACK: clean up bottom right corner
    clean_binary_image[-2:, -2:] = 0
    return clean_binary_image

def get_top_contour(bin_image):
    horizontal_edges = bin_image[1:, :] - bin_image[:-1, :]

    (rows, cols) = horizontal_edges.shape;

    top_contour = np.zeros(cols)

    # TODO(vivek): find a way use anti-aliasing to get sub-pixel accuracy. 
    for c in range(cols):
        for r in range(rows):
            if horizontal_edges[r, c] == 255:
                top_contour[c] = r;
                break

    # HACK: clean up right end of top contour
    top_contour[-2:] = 0
    return top_contour

def contour_to_world_points(contour, cont_range, size, heights, ref_point):
    (height, width) = size
    
    world_points = []
    for i in range(cont_range[0], cont_range[1]):
        iy = (height/2) - contour[i]
        ix = (width/2) - i

        ih_ref = heights[ref_point]
        wz_ref = 1;
        wz = (ih_ref * wz_ref) / heights[i]
        wx = wz * ix / focal;
        wy = wz * iy / focal;
        world_points.append([wx, wy, wz])

    return np.array(world_points)

def world_to_source(wx, wy, wz):
    tix = focal * wx / wz;
    tiy = focal * wy / wz;
    return tix, tiy

def find_ref_point_index(contour):
    edge_boundaries = np.where(abs(contour[1:] - contour[:-1]) > 5)[0]
    ref_point_index = edge_boundaries[0]+1;
    return ref_point_index

def gen_dist_array(wx, wz, x_interval):

    delta_x = x_interval[1] - x_interval[0]
    delta_z_arr = fWxToWz(x_interval[1:]) - fWxToWz(x_interval[:-1])

    distance = [0]
    for delta_z in delta_z_arr:
        dist = np.sqrt(delta_x * delta_x + delta_z * delta_z)
        distance.append(distance[-1] + dist)

    return np.array(distance)


file_path = sys.argv[1]
focal = float(sys.argv[2])

image = cv2.imread(file_path)
binary_image = create_binary_image(image)
top_contour = get_top_contour(binary_image)
ref_point_index = find_ref_point_index(top_contour)

top_contour_range = np.where(abs(top_contour[1:] - top_contour[:-1]) > 5)[0] + 1

(image_height, image_width) = binary_image.shape
height_map = (image_height/2) - top_contour

world_points = contour_to_world_points(top_contour, top_contour_range, (image_height, image_width), height_map, ref_point_index)
# TODO(vivek): check that all Y values are the same

wx = world_points[:, 0]
wz = world_points[:, 2]

# ** ALERT ** wx might be flipped 

fWxToWz = interp1d(wx[::-1], wz, kind='cubic')

x_interval = np.linspace(min(wx), max(wx), num=100, endpoint=True)

distance_arr = gen_dist_array(wx, wz, x_interval)

width = distance_arr[-1]
height = world_points[0, 1]

fDistToWx = interp1d(distance_arr, x_interval)
fDistToWz = interp1d(distance_arr, fWxToWz(x_interval))

aspect = height / width
output_width = 500
output_height = int(aspect * output_width)

print 'Begin Unwarping'

output_image = np.zeros((output_height, output_width, 3))
for oy in range(len(output_image)):
    for ox in range(len(output_image[0])):
        d = (ox / float(output_width)) * width
        v = (oy / float(output_height)) * height
        
        wx = fDistToWx(d)
        wy = v
        wz = fDistToWz(d)

        ix, iy = world_to_source(wx, wy, wz)

        image_x = (image_width/2) - ix
        image_y = (image_height/2) - iy

        pixel = cv2.getRectSubPix(image, (1, 1), (image_x, image_y))
        output_image[output_height - oy - 1, output_width - ox - 1] = pixel[0, 0] / 255.0


cv2.imwrite('./test_focal_length/'+str(focal)+'.png', output_image * 255.0)





