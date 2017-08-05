import cv2
import numpy as np

image = cv2.imread('./warped-puppy.png')


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


res, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

# Size kernel to fit image!
kernel = np.ones((3,3),np.uint8)
clean_binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# HACK: clean up bottom right corner
clean_binary_image[-2:, -2:] = 0

horizontal_edges = clean_binary_image[1:, :] - clean_binary_image[:-1, :]

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

edge_boundaries = np.where(abs(top_contour[1:] - top_contour[:-1]) > 5)[0]

cropped_top_contour = top_contour[edge_boundaries[0]+1:edge_boundaries[1]+1]
#print cropped_top_contour

ref_point_index = edge_boundaries[0]+1;

image_height = rows;
image_width = cols;

# ASSUMPTION: bottom contour lines up with a horizontal plane cutting through pinhole.

heights = (image_height/2) - top_contour
cropped_heights = (image_height/2) - cropped_top_contour

out, contours, hier = cv2.findContours(clean_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#cv2.drawContours(image, contours, -1, (0,255,0), 3)
bounding_box = cv2.boundingRect(contours[0])
p1 = bounding_box[0:2]
p2 = (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])
#cv2.rectangle(image, p1, p2, (255, 0, 0))

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

point_cloud = []

for y in range(bounding_box[3]):
    for x in range(bounding_box[2]):
        ix = x + bounding_box[0]
        iy = y + bounding_box[1]
        #image[iy, ix] = (255, 0, 0)

        wz = worldZ(ix, iy)
        wx = worldX(ix, iy)
        wy = worldY(ix, iy)
        color = image[iy, ix];
        point = np.array([wx, wy, wz])
        point_cloud.append([point, color]);

np_point_cloud = np.array(point_cloud);

x_min = min(np_point_cloud[:, 0, 0])
x_max = max(np_point_cloud[:, 0, 0])

y_min = min(np_point_cloud[:, 0, 1])
y_max = max(np_point_cloud[:, 0, 1])

print x_min, x_max, y_min, y_max

x_range = (x_max - x_min)
y_range = (y_max - y_min)
aspect_ratio = x_range / y_range

fixed_width = 500
scaled_height = int(np.ceil(500 / aspect_ratio))

print fixed_width, scaled_height
rectifiedImage = np.zeros((scaled_height + 1, fixed_width + 1, 3))

for (point, color) in point_cloud:
    rx = int(((point[0] - x_min) / (x_range)) * fixed_width)
    ry = int(((point[1] - y_min) / (y_range)) * scaled_height)
    #print color
    #print rectifiedImage[ry, rx].shape
    rectifiedImage[ry, rx] = np.array(color) / 255.0


cv2.imshow('original image', image)
cv2.imshow('rectified image', rectifiedImage)
cv2.waitKey(0);





#image[10, :] = (0, 0, 255)
#cv2.imshow('', image);
#cv2.waitKey(0);





