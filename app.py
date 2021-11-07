# Import modules
import cv2
import argparse
import numpy as np
from PIL import Image
import mediapipe as mp
from os.path import splitext
from img_utils import mls_rigid_deformation


# Functions
def resize_face_part(input_image, old_points, new_points):
    """
    Resizing the face part.
  
    Parameters:
    input_image (ndarray): array of input image
    old_points (list): list of marked points before resizing
    new_points (list): list of marked points after resizing
  
    Returns:
    ndarray: Array of resized image
  
    """
    p = np.array(old_points)
    q = np.array(new_points)
        
    height, width, _ = input_image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    drigid = mls_rigid_deformation(vy, vx, p, q)
    rigid = drigid.astype(np.float32)
    new_img = cv2.remap(input_image, rigid[1], rigid[0], interpolation=cv2.INTER_CUBIC)
    return new_img
    

def landmarks(image):
    """
    Extract the face landmarks.
  
    Parameters:
    image (ndarray): array of input image
  
    Returns:
    im_height: Height of image
    im_width: Width of image
    points: Face landmarks
  
    """
    im_height, im_width, im_channel = image.shape
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    points = []
    for i,res in enumerate(results.multi_face_landmarks[0].ListFields()[0][1]):
        x_coor = int(res.x * im_width)
        y_coor = int(res.y * im_height)    
        points.append([x_coor, y_coor])
        
    return im_height, im_width, points


def add_margin(input_image, image_width, image_height,
            points_1, points_2, mark_points):
    """
    Paste the resized image on the blank background(with the size of original image).
  
    Parameters:
    input_image (ndarray): array of input image
    image_width (int): width of the original image
    im_height (int): height of the original image
    points_1 (list): points in the original image
    points_2 (list): the same points in the resized image
    mark_points (list): the landmarks

    Returns:
    ndarray: Array of resized image with black background
  
    """    
    le_x_center1 = int((points_1[mark_points[0]][0] + points_1[mark_points[1]][0]) / 2)
    le_y_center1 = int((points_1[mark_points[2]][1] + points_1[mark_points[3]][1]) / 2)

    le_x_center2 = int((points_2[mark_points[0]][0] + points_2[mark_points[1]][0]) / 2)
    le_y_center2 = int((points_2[mark_points[2]][1] + points_2[mark_points[3]][1]) / 2)

    c_point_x = le_x_center1 - le_x_center2
    c_point_y = le_y_center1 - le_y_center2

    im1 = Image.new('RGB', (image_width, image_height))
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    im2 = Image.fromarray(img)

    back_im = im1.copy()
    back_im.paste(im2, (c_point_x, c_point_y))
    return np.array(back_im)


def resize_image(image, scale_percent):
    """
    Image resizing.
  
    Parameters:
    image (ndarray): array of input image
    scale_percent (int): percent of scaling

    Returns:
    ndarray: Array of resized image

    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    scaled_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return scaled_image


def main(input_path, scale_factor):
    """
    Left eye and nose resizing.
  
    Parameters:
    input_path (ndarray): array of input image
    scale_factor (int): percent of scaling

    Returns:
    A deformed image: Save the result in the same directory of input image

    """
    # mark-points
    l_eye_points = [463, 414, 286, 258, 257, 259, 260, 467, 359,
                   255, 339, 254, 253, 252, 256, 341]

    nose_points = [114, 343, 217, 437, 126, 355, 209, 429,
                  129, 358, 98, 327, 167, 393, 164]

    input_scale = 100 + scale_factor

    input_image = cv2.imread(input_path)

    ## Left Eye
    # Resize the image
    resized_for_eye = resize_image(input_image, input_scale)

    # Extract the input image landmarks
    im_height1, im_width1, points1 = landmarks(input_image)

    # Extract the scaled image landmarks
    im_height2, im_width2, points2 = landmarks(resized_for_eye)

    # Merge for left eye
    merged_image_leye = add_margin(resized_for_eye, im_width1, im_height1,
                                    points1, points2, [463,359,257,253])

    # Extract the merged image landmarks
    im_height3, im_width3, points3 = landmarks(merged_image_leye)

    # old and new points for face part ersizing
    oldi = []
    mod_points = []
    oldi.append(points1[234])
    oldi.append(points1[454])
    mod_points.append(points1[234])
    mod_points.append(points1[454])
    for num in l_eye_points:
        oldi.append(points1[num])
        mod_points.append(points3[num])
        
    # Modify left eye
    eye_mod_image = resize_face_part(input_image, old_points=oldi, new_points=mod_points)

    
    ## Nose
    # Resize the eye modified image
    resized_for_nose = resize_image(eye_mod_image, input_scale)

    # Extract the eye modified image landmarks
    im_height1, im_width1, points1 = landmarks(eye_mod_image)

    # Extract the scaled eye modified image landmarks
    im_height2, im_width2, points2 = landmarks(resized_for_nose)

    # Merge for left eye (for minifying)
    merged_image_nose = add_margin(resized_for_nose, im_width1, im_height1,
                                    points1, points2, [6,6,6,2])

    # Extract the merged image landmarks
    im_height3, im_width3, points3 = landmarks(merged_image_nose)

    # old and new points for face part ersizing
    oldi = []
    mod_points = []
    oldi.append(points1[21])
    oldi.append(points1[389])
    oldi.append(points1[136])
    oldi.append(points1[365])
    mod_points.append(points1[21])
    mod_points.append(points1[389])
    mod_points.append(points1[136])
    mod_points.append(points1[365])
    for num in nose_points:
        oldi.append(points1[num])
        mod_points.append(points3[num])
        
    # Resize left eye
    ult_image = resize_face_part(eye_mod_image, old_points=oldi, new_points=mod_points)

    # Save the ultimate result
    cv2.imwrite(f"{splitext(input_path)[0]}-output.jpg", ult_image)



if __name__ == "__main__":
    app_parser = argparse.ArgumentParser(description='Rescaling the left eye and the nose of a face image')
    app_parser.add_argument('--image', help='The image path', type=str, required=True)
    app_parser.add_argument('--threshold', help='The scale in percent ("+": to enlarge, "-": to minify)',
                            type=int, required=True)

    args = vars(app_parser.parse_args())

    try:
        main(input_path=args['image'], scale_factor=args['threshold'])
        print('\nDone!')
    except:
        print('\n--> The face not found; Check if the image has face or the input threshold is very small')
