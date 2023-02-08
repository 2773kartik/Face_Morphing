'''
Kartik Tiwari
- Face Morphing program that takes two images as input and generates n frams (specified by user) to perform face morphing 
Using delaunay triangulation. Please ensure that the dimensions of both images in input are same.
"shape_predictor_68_face_landmarks.dat" used to detect 68 landmark points on face
'''

from imutils import face_utils  #For face points mapping
import dlib                     #For the model to detect landmarks automatically
# To install dlib, first pip install cmake, then pip install dlib
import cv2                      #OpenCV library
import os                       #For saving images and video in a path
import numpy             #Numpy library

#To return indices of triangles in image
def find_i(point, tr):
    cnt = 0
    for index in tr:
        if point == index:
            return cnt
        cnt += 1

#Delaunay triangulations
def triangulate(img, point):
    #Subdiv class instance
    subs = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))
    for p in point:
        subs.insert(p)
    #Returns triangles vertices
    return subs.getTriangleList()

#Combine input and output via affine transform
def combineResult(img1, img2, img, t1, t2, t, alpha):
    rect1 = cv2.boundingRect(numpy.float32([t1]))
    rect2 = cv2.boundingRect(numpy.float32([t2]))
    rect = cv2.boundingRect(numpy.float32([t]))
    t1_cropped, t2_cropped, t_cropped = [],[],[]

    for i in range(3):
        t1_cropped.append(((t1[i][0] - rect1[0]), (t1[i][1] - rect1[1])))
        t2_cropped.append(((t2[i][0] - rect2[0]), (t2[i][1] - rect2[1])))
        t_cropped.append(((t[i][0] - rect[0]), (t[i][1] - rect[1])))

    mask = numpy.zeros((rect[3], rect[2], 3), dtype=numpy.float32)
    cv2.fillPoly(mask, [numpy.int32(t_cropped)], (1.0, 1.0, 1.0), 16)

    img1_roi = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    img2_roi = img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]

    size = (rect[2], rect[3])
    warp1 = cv2.getAffineTransform(numpy.float32(t1_cropped), numpy.float32(t_cropped))
    warp_img1 = cv2.warpAffine(img1_roi, warp1, size)
    warp2 = cv2.getAffineTransform(numpy.float32(t2_cropped), numpy.float32(t_cropped))
    warp_img2 = cv2.warpAffine(img2_roi, warp2, size)

    result_roi = (1. - alpha) * warp_img1 + alpha * warp_img2

    img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] * (1-mask) + result_roi * mask


#Mark the landmarks and return coordinates
def get_points(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        shaped = predictor(gray, rect)
        shaped = face_utils.shape_to_np(shaped)
    h, w, c = img.shape
    triangle = []
    for coord in shaped:
        triangle.append((int(coord[0]),int(coord[1])))
    triangle.extend([(0,0), (0,h-1), (w-1,h-1), (w-1,0)])
    return triangle

if __name__ == '__main__':
    frames = int(input("How many frames to generate?")) #20 recommended (optimal)
    fps = frames/2
    #Open the model to detect face
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    #Path of the two images to be morphed
    image = cv2.imread('a.jpg')
    image1 = cv2.imread('b.jpg')

    tr1 = get_points(image)
    tr2 = get_points(image1)

    dlny1 = triangulate(image, tr1)

    tri_index = []
    for t in dlny1:
        pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
        # Creating a combined point of (x,y)
        add = [(find_i(pt1, tr1), find_i(pt2, tr1), find_i(pt3, tr1))]
        tri_index.extend(add)

    #Create a video from the generated frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = image.shape
        # Docs : https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html
    videoWriter = cv2.VideoWriter(".\output_video\Morph.mp4", fourcc, fps, (w, h))
    path = ".\output_images"
    for k in range(0, frames + 1):
        alpha, tr_Middle = k / frames, []

        for i in range(len(tr1)):
            x = int(((1-alpha) * tr1[i][0]) + (alpha * tr2[i][0]))
            y = int(((1-alpha) * tr1[i][1]) + (alpha * tr2[i][1]))
            tr_Middle.append((x, y))

            # Creating a image with as same as shape of image having zeroes
        imgMorph = numpy.zeros(image.shape, dtype=image1.dtype)

        for j in range(len(tri_index)):

            x, y, z = tri_index[j][0], tri_index[j][1], tri_index[j][2]
            t1 = [tr1[x], tr1[y], tr1[z]]
            t2 = [tr2[x], tr2[y], tr2[z]]
            t = [tr_Middle[x], tr_Middle[y], tr_Middle[z]]

            # Doing morphing with passed triangles and images
            combineResult(image, image1, imgMorph, t1, t2, t, alpha)

        cv2.imshow("Morphed Face", numpy.uint8(imgMorph))
        cv2.imwrite(os.path.join(path , str(k)+".jpg"), imgMorph)
        videoWriter.write(imgMorph)
        cv2.waitKey(50)
    videoWriter.release()
    cv2.destroyAllWindows()