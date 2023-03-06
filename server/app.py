import cv2
import numpy as np
import os
import dlib
import imageio

path="C:/Users/Devanshu/Desktop/My folder/Proper JS/Project Morphing/server/uploads"
IMAGES=[]
for p in os.listdir(path):
    IMAGES.append(cv2.imread(os.path.join(path, p)))
img1, img2=IMAGES[:2]

if(img1.shape[0]>500 or img1.shape[1]>500):
    if img1.shape[0]>img1.shape[1]:
        scaleimg1 = float(500/img1.shape[0])
    else :scaleimg1 = float(500/img1.shape[1])
    img1 = cv2.resize(img1,None,fx=scaleimg1,fy=scaleimg1,interpolation=cv2.INTER_AREA)

# print(img1.shape) 

# Function for extracting the index from the array
def extracting(array):
    index = None
    for i in array[0]:
        index = i
        break
    return index

def apply_affine_transform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morph_triangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    # try : 
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask
    # except: 
    #     print("ERROR!!")

def calculate_margin_help(img1,img2):
    size1 = img1.shape
    size2 = img2.shape
    diff0 = abs(size1[0]-size2[0])//2
    diff1 = abs(size1[1]-size2[1])//2
    avg0 = (size1[0]+size2[0])//2
    avg1 = (size1[1]+size2[1])//2

    return [size1,size2,diff0,diff1,avg0,avg1]

def crop_image(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)

    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        scale0 = float(size1[0])/size2[0]
        scale1 = float(size1[1])/size2[1]
        # print(scale0)
        # print(scale1)
        if(scale0 > scale1):
            res = cv2.resize(img2,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img2,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(img1,res)

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        scale0 = float(size2[0])/size1[0]
        scale1 = float(size2[1])/size1[1]
        if(scale0 > scale1):
            res = cv2.resize(img1,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img1,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(res,img2)

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,diff1:avg1]]
        # return [img1[diff0:avg0,:],img2[:,-diff1:avg1]]
    
    else:
        return [img1[:,diff1:avg1],img2[diff0:avg0,:]]

def crop_image_help(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)
    
    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        # return [img1,img2[:,:]]
        # print(diff0, " yo ", avg0 , " " , diff1 ," " , avg1)
        return [img1,img2[diff0:avg0,diff1:avg1]]

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        # return [img1[:,:],img2]
        return [img1[diff0:avg0,diff1:avg1],img2]

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,diff1:avg1]]

    else:
        return [img1[:,diff1:avg1],img2[diff0:avg0,:]]


# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# read the images
# img1 = cv2.imread("face3.jpeg")
# img2 = cv2.imread("face1.jpeg")
img1,img2 = crop_image(img1,img2)
# Convert images into grayscale
gray1 = cv2.cvtColor(src=img1, code=cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(src=img2, code=cv2.COLOR_BGR2GRAY)

# # Convert Mat to float data type
# img1 = np.float32(img1)
# img2 = np.float32(img2)

# Use detector to find landmarks for image 1
faces = detector(gray1)

landmark_points_1 = []
count =0
for face in faces:
    count+=1

    # Create landmark object
    landmarks = predictor(image=gray1, box=face)

    # Loop through all the points
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points_1.append((x,y))

        # Draw a circle
        # cv2.circle(img=img1, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
if count == 0 or count > 1 : 
    if count == 0:
        print("No face Detected in image 1")
    else : print("More than one face detected in image 1")
    exit()

#inserting corner points of the image 1
landmark_points_1.append((0,0))
landmark_points_1.append((0,img1.shape[0]-1))
temp1 = int(img1.shape[0]/2)
landmark_points_1.append((0,temp1))
temp2 = int(img1.shape[1]/2)
landmark_points_1.append((img1.shape[1]-1,0))
landmark_points_1.append((temp2,0))
landmark_points_1.append((img1.shape[1]-1,img1.shape[0]-1))
landmark_points_1.append((img1.shape[1]-1,temp1-1))
landmark_points_1.append((temp2-1,img1.shape[0]-1))

# Use detector to find landmarks for image 2
faces = detector(gray2)
landmark_points_2 = []

count=0
for face in faces:
    # Create landmark object
    count +=1
    landmarks = predictor(image=gray2, box=face)

    # Loop through all the points
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points_2.append((x,y))

        # Draw a circle
        # cv2.circle(img=img2, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
if count == 0 or count > 1 : 
    if count == 0:
        print("No face Detected in image 2")
    else : print("More than one face detected in image 2")
    exit()


#inserting corner points of the image 2
temp1 = int(img2.shape[0]/2)
temp1 = int(img2.shape[1]/2)
landmark_points_2.append((0,0))
landmark_points_2.append((0,img2.shape[0]-1))
landmark_points_2.append((0,temp1))
landmark_points_2.append((img2.shape[1]-1,0))
landmark_points_2.append((temp2,0))
landmark_points_2.append((img2.shape[1]-1,img2.shape[0]-1))
landmark_points_2.append((img2.shape[1]-1,temp1))
landmark_points_2.append((temp2,img2.shape[0]-1))

points_1 = np.array(landmark_points_1,np.int32)
convexhull_1 = cv2.convexHull(points_1)

points_2 = np.array(landmark_points_2,np.int32)
convexhull_2 = cv2.convexHull(points_2)

# Delaunay triangulation for image 1
rect = cv2.boundingRect(convexhull_1)
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(landmark_points_1)
triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype=np.int32)
for t in triangles:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])
    # cv2.line(img1, pt1, pt2, (0, 0, 255), 2)
    # cv2.line(img1, pt2, pt3, (0, 0, 255), 2)
    # cv2.line(img1, pt1, pt3, (0, 0, 255), 2)

triangle_indexes = []

for t in triangles:
    # Initializing the points 
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])

    # Extracting the points and axis = 1 because if axis != 1it will give
    # all the points having the same x or y 
    index_pt1 = np.where((points_1 == pt1).all(axis=1))
    index_pt1 = extracting(index_pt1)
    index_pt2 = np.where((points_1 == pt2).all(axis=1))
    index_pt2 = extracting(index_pt2)
    index_pt3 = np.where((points_1 == pt3).all(axis=1))
    index_pt3 = extracting(index_pt3)

    # Appending triangles coordinates
    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangle = [index_pt1, index_pt2, index_pt3]
        triangle_indexes.append(triangle)

# Triangulation of the second face, from the first face delaunay triangulation
for triangle_index in triangle_indexes:
    pt1 = landmark_points_2[triangle_index[0]]
    pt2 = landmark_points_2[triangle_index[1]]
    pt3 = landmark_points_2[triangle_index[2]]


# Allocate space for final output
imgMorph = []
for i in range(0,80):
    # imgMorph = np.zeros(img1.shape, dtype = img1.dtype)
    tempimg = np.zeros(img1.shape, dtype = img1.dtype)
    points = []
    alpha = (i+1)/80

    # Compute weighted average point coordinates
    for i in range(0, len(points_1)):
        x = ( 1 - alpha ) * points_1[i][0] + alpha * points_2[i][0]
        y = ( 1 - alpha ) * points_1[i][1] + alpha * points_2[i][1]
        points.append((x,y))

    for triangle in triangle_indexes :
        x,y,z = triangle[0],triangle[1],triangle[2]
        
        x = int(x)
        y = int(y)
        z = int(z)
        
        t1 = [points_1[x], points_1[y], points_1[z]]
        t2 = [points_2[x], points_2[y], points_2[z]]
        t = [ points[x], points[y], points[z] ]

        # Morph one triangle at a time.
        morph_triangle(img1, img2, tempimg, t1, t2, t, alpha)
        
    tempimg_rgb = cv2.cvtColor(tempimg, cv2.COLOR_BGR2RGB)
    imgMorph.append(tempimg_rgb)


imageio.mimwrite('./downloads/result.gif',imgMorph,fps=20)


def calculate_margin_help(img1,img2):
    size1 = img1.shape
    size2 = img2.shape
    diff0 = abs(size1[0]-size2[0])//2
    diff1 = abs(size1[1]-size2[1])//2
    avg0 = (size1[0]+size2[0])//2
    avg1 = (size1[1]+size2[1])//2

    return [size1,size2,diff0,diff1,avg0,avg1]

def crop_image(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)

    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        scale0 = size1[0]/size2[0]
        scale1 = size1[1]/size2[1]
        if(scale0 > scale1):
            res = cv2.resize(img2,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img2,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(img1,res)

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        scale0 = size2[0]/size1[0]
        scale1 = size2[1]/size1[1]
        if(scale0 > scale1):
            res = cv2.resize(img1,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img1,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(res,img2)

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        # if dif00
        return [img1[diff0:avg0,:],img2[:,diff1:avg1]]
    
    else:
        return [img1[:,diff1:avg1],img2[diff0:avg0,:]]

def crop_image_help(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)
    
    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        return [img1,img2[diff0:avg0,diff1:avg1]]

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        return [img1[diff0:avg0,diff1:avg1],img2]

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,diff1:avg1]]

    else:
        return [img1[:,diff1:avg1],img2[diff0:avg0,:]]