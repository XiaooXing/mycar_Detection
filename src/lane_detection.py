import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
# from moviepy.editor import VideoFileClip

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold

# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 40
max_line_gap = 20


def roi_mask(img, vertices):
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        mask_color = (255,) * channel_count
    else:
        mask_color = 255

    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_roi(img, vertices):
    cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    draw_lanes(line_img, lines)
    return line_img


def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return img

    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])
    right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])

    cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
    cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)


def clean_lines(lines, threshold):
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


def calc_lane_vertices(point_list, ymin, ymax):
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))

    return [(xmin, ymin), (xmax, ymax)]


def process_an_image(img,name,image_root):
    roi_vtx = np.array([[(0, img.shape[0]), (460, 325), (520, 325), (img.shape[1], img.shape[0])]])

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    roi_edges = roi_mask(edges, roi_vtx)
    line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)
    res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    plt.figure()
    plt.imshow(gray, cmap='gray')
    # plt.savefig('../resources/gray.png', bbox_inches='tight')
    plt.savefig(os.path.join(image_root,'gray',name),bbox_inches = 'tight')
    plt.figure()
    plt.imshow(blur_gray, cmap='gray')
    # plt.savefig('../resources/blur_gray.png', bbox_inches='tight')
    plt.savefig(os.path.join(image_root,'blur_gray',name),bbox_inches = 'tight')
    plt.figure()
    plt.imshow(edges, cmap='gray')
    # plt.savefig('../resources/edges.png', bbox_inches='tight')
    plt.savefig(os.path.join(image_root,'edges',name),bbox_inches = 'tight')

    plt.figure()
    plt.imshow(roi_edges, cmap='gray')
    # plt.savefig('../resources/roi_edges.png', bbox_inches='tight')
    plt.savefig(os.path.join(image_root,'roi_edges',name),bbox_inches = 'tight')
    plt.figure()
    plt.imshow(line_img, cmap='gray')
    # plt.savefig('../resources/line_img.png', bbox_inches='tight')
    plt.savefig(os.path.join(image_root,'line_img',name),bbox_inches = 'tight')
    plt.figure()
    plt.imshow(res_img)
    # plt.savefig('../resources/res_img.png', bbox_inches='tight')
    plt.savefig(os.path.join(image_root,'res_img',name),bbox_inches = 'tight')
    # plt.show()
    # return res_img


# img = mplimg.imread("../resources/lane.jpg")
image_root = './image_video1'
name_list = os.listdir(os.path.join(image_root,'image'))
os.makedirs(os.path.join(image_root,'gray'),exist_ok=True)
os.makedirs(os.path.join(image_root,'blur_gray'),exist_ok=True)
os.makedirs(os.path.join(image_root,'edges'),exist_ok=True)
os.makedirs(os.path.join(image_root,'roi_edges'),exist_ok=True)
os.makedirs(os.path.join(image_root,'line_img'),exist_ok=True)
os.makedirs(os.path.join(image_root,'res_img'),exist_ok=True)
for name in name_list:
    img = cv2.imread(os.path.join(image_root,'image',name))
    process_an_image(img,name,image_root)
print('successfully processed!')
# img = cv2.imread('image_video1/1.jpg')
# process_an_image(img)

# output = '../resources/video_1_sol.mp4'
# clip = VideoFileClip("../resources/video_1.mp4")
# out_clip = clip.fl_image(process_an_image)
# out_clip.write_videofile(output, audio=False)
# video_File = 'resources/video_1.mp4'
# outputFile = './image_video1'
# os.makedirs(outputFile,exist_ok=True)
# vc = cv2.VideoCapture(video_File)
# c = 1
# if vc.isOpened():
#     rval, frame = vc.read()
# else:
#     print('openerror!')
#     rval = False

# timeF = 10
# while rval:
#     print(1)

#     rval, frame = vc.read()
#     if c % timeF ==0:
#         print(2)
#         cv2.imwrite(os.path.join(outputFile,str(int(c/timeF))+'.jpg'),frame)
#     c+=1
#     cv2.waitKey(1)
# vc.release()
# print('run successfully')