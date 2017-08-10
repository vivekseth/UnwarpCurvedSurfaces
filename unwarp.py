import sys
import cv2
import numpy as np
import scipy as scp
from scipy.interpolate import griddata

def binary_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

    # Size kernel to fit image!
    kernel = np.ones((3,3),np.uint8)
    clean_binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # HACK: clean up bottom right corner
    clean_binary_image[-2:, -2:] = 0
    return clean_binary_image

def top_contour(bin_image):
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

def bounding_box(bin_image):
    out, contours, hier = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_box = cv2.boundingRect(contours[0])
    return bounding_box

image = cv2.imread('./screenshot2.png')
bin_image = binary_image(image)
top_con = top_contour(bin_image)
box = bounding_box(bin_image)

edge_boundaries = np.where(abs(top_con[1:] - top_con[:-1]) > 5)[0]
ref_point_index = edge_boundaries[0]+1;

(image_height, image_width) = bin_image.shape

# ASSUMPTION: bottom contour lines up with a horizontal plane cutting through pinhole.
heights = (image_height/2) - top_con


# Visualizations: 
# p1 = bounding_box[0:2]
# p2 = (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])
# cv2.drawContours(image, contours, -1, (0,255,0), 3)
# cv2.rectangle(image, p1, p2, (255, 0, 0))
# for p in contours[0]:
#     x = p[0][0]
#     y = p[0][1]
#     image[y, x] = (255, 0, 0)


# HACK: just guessing for now.
focal = 64;

def worldZ(ix, iy):
    ih_ref = heights[ref_point_index]
    wz_ref = 1;
    return (ih_ref * wz_ref) / heights[ix]

def worldX(ix, iy):
    wz = worldZ(ix, iy);
    pinholeX = ix - (image_width/2)
    return wz * pinholeX / focal;

def worldY(ix, iy):
    wz = worldZ(ix, iy);
    pinholeY = iy - (image_height/2)
    return wz * pinholeY / focal;

#########################################
# IMAGES MUST BE ACCESSED IN Y, Z ORDER #
#########################################

def create_point_cloud(image, box):
    point_cloud = []

    for y in range(box[3]):
        for x in range(box[2]):
            ix = x + box[0]
            iy = y + box[1]
            #image[iy, ix] = (255, 0, 0)

            wz = worldZ(ix, iy)
            wx = worldX(ix, iy)
            wy = worldY(ix, iy)
            color = image[iy, ix];
            point = np.array([wx, wy, wz])
            point_cloud.append([point, color]);

    np_point_cloud = np.array(point_cloud);
    return np_point_cloud

point_cloud = create_point_cloud(image, box)


def point_cloud_range(point_cloud):
    x_min = min(point_cloud[:, 0, 0])
    x_max = max(point_cloud[:, 0, 0])

    y_min = min(point_cloud[:, 0, 1])
    y_max = max(point_cloud[:, 0, 1])

    return x_min, x_max, y_min, y_max

x_min, x_max, y_min, y_max = point_cloud_range(point_cloud)

def interpolation_points(x_min, x_max, y_min, y_max):
    x_range = (x_max - x_min)
    y_range = (y_max - y_min)

    # 100x100 grid of x and y values to interpolate
    x_step = (x_range / 100)
    y_step = (y_range / 100)
    grid_x, grid_y = np.mgrid[x_min+x_step:x_max+x_step:(x_range / 100), y_min+y_step:y_max:(y_range / 100)]
    return grid_x, grid_y

grid_x, grid_y = interpolation_points(x_min, x_max, y_min, y_max)

def interpolate_point_cloud_1(point_cloud, grid_x, grid_y):
    interp_points = point_cloud[:, 0, 0:2]
    interp_values_z = point_cloud[:, 0, 2]
    interp_values_b = point_cloud[:, 1, 0]
    interp_values_g = point_cloud[:, 1, 1]
    interp_values_r = point_cloud[:, 1, 2]

    print 'BBBBB', interp_values_b

    grid_z = griddata(interp_points, interp_values_z, (grid_x, grid_y), method='cubic')
    grid_b = griddata(interp_points, interp_values_b, (grid_x, grid_y), method='cubic')
    grid_g = griddata(interp_points, interp_values_g, (grid_x, grid_y), method='cubic')
    grid_r = griddata(interp_points, interp_values_r, (grid_x, grid_y), method='cubic')

    return grid_z, grid_b, grid_g, grid_r

grid_z, grid_b, grid_g, grid_r = interpolate_point_cloud_1(point_cloud, grid_x, grid_y)

def dist_arr_for_y(y_index):
    dist_arr = np.zeros(len(grid_x))
    for i in range(len(grid_x) - 1):
        x = grid_x[i][y_index]
        y = grid_y[i][y_index]
        z = grid_z[i][y_index]
        nx = grid_x[i+1][y_index]
        ny = grid_y[i+1][y_index]
        nz = grid_z[i+1][y_index]
        dx = nx - x;
        dz = nz - z;
        local_dist = np.sqrt(dx*dx + dz*dz);
        dist_arr[i+1] = dist_arr[i] + local_dist;
    return dist_arr

all_dist_arr = []
for i in range(len(grid_y[0])):
    dist_arr = dist_arr_for_y(i)
    max_dist = max(dist_arr);
    dist_arr[np.isnan(dist_arr)] = 0
    all_dist_arr.append(dist_arr)

np_all_dist_arr = np.array(all_dist_arr).T

print grid_y.shape
print np_all_dist_arr.shape
print grid_b.shape

x_range = (x_max - x_min)
y_range = (y_max - y_min)

avg_row_dist = np.mean(np_all_dist_arr[-1])
print avg_row_dist, y_range

aspect_ratio = avg_row_dist / y_range

fixed_width = 500
scaled_height = int(np.ceil(500 / aspect_ratio))

#print fixed_width, scaled_height
rectifiedImage = np.zeros((scaled_height, fixed_width, 3))
print rectifiedImage.shape

# Create "point cloud" for Yloc, distance -> color

color_point_cloud = []

for i in range(len(grid_b)):
    for j in range(len(grid_b[0])):
        y = grid_y[i][j]
        d = np_all_dist_arr[i][j]
        b = grid_b[i][j]
        g = grid_g[i][j]
        r = grid_r[i][j]
        color_point_cloud.append([y, d, b, g, r])

np_color_point_cloud = np.array(color_point_cloud)
print '$$', np_color_point_cloud.shape

img_grid_y, img_grid_x = np.mgrid[0:scaled_height:1, 0:fixed_width:1]

print np_color_point_cloud[0, 1]

interp_points_yd = np_color_point_cloud[:, (0, 1)]
interp_values_yd_b = np_color_point_cloud[:, 2]
interp_values_yd_g = np_color_point_cloud[:, 3]
interp_values_yd_r = np_color_point_cloud[:, 4]

y_vals = interp_points_yd[:, 0]
y_max = max(y_vals)
y_min = min(y_vals)
y_range = y_max - y_min
scaled_y_vals = ((y_vals - y_min) / y_range) * scaled_height

d_vals = interp_points_yd[:, 1]
d_max = max(d_vals)
d_min = min(d_vals)
d_range = d_max - d_min
scaled_d_vals = ((d_vals - d_min) / d_range) * fixed_width

scaled_yd = np.zeros(interp_points_yd.shape)
scaled_yd[:, 0] = scaled_y_vals
scaled_yd[:, 1] = scaled_d_vals


print interp_points_yd
print ''
print interp_values_yd_b
print ''
print img_grid_y
print ''
print img_grid_x

im_grid_b = griddata(scaled_yd, interp_values_yd_b, (img_grid_y, img_grid_x), method='cubic')
im_grid_g = griddata(scaled_yd, interp_values_yd_g, (img_grid_y, img_grid_x), method='cubic')
im_grid_r = griddata(scaled_yd, interp_values_yd_r, (img_grid_y, img_grid_x), method='cubic')


rectifiedImage[:, :, 0] = im_grid_b #/ 255.0
rectifiedImage[:, :, 1] = im_grid_g #/ 255.0
rectifiedImage[:, :, 2] = im_grid_r #/ 255.0

cv2.imwrite('./rectifiedImage.png', rectifiedImage)

# cv2.imshow('TEST', rectifiedImage)
# cv2.waitKey(0);

