import cv2
import numpy as np
import math

cap = cv2.VideoCapture(4)
right = cv2.imread("/home/yagna-rao15/Code/Rover/Images/right.jpg", cv2.IMREAD_GRAYSCALE)
left = cv2.imread("/home/yagna-rao15/Code/Rover/Images/left.jpg", cv2.IMREAD_GRAYSCALE)
threshold=0.8

def preprocess(img):
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img, (5, 5), 1)
    img_canny = cv2.Canny(img, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])
        
def convert_to_binary(frame):
    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)
    # _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # normalized_image = binary_image.astype(np.uint8)
    return blurred_image


def direction(approx):
    if(len(approx)>=7):
        l=r=0
        for a in approx:
            if(a[0,0]>arrow_tip[0]):
                l+=1
            if(a[0,0]<arrow_tip[0]):
                r+=1

        if(l>4):
            return -1
        if(r>4):
            return 1
        else:
            return 0
    else: return 0



# def mean(approx):
#     sumx=sumy=0
#     for a in approx:
#         sumx+=a[0,0]
#         sumy+=a[0,1]
#     centroid=[]
#     centroid.append((sumx/7)-320)
#     centroid.append((sumy/7)-240)
#     return centroid

def multi_scale_template_matching(image, template):
    max=-1
    maxi_loc=-1
    max_scale=-1
    for scale in np.linspace(0.08,0.5,20):
        scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
        result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_loc = cv2.minMaxLoc(result)
        if max_value>=threshold and max_value>max:
            max=max_value
            maxi_loc=max_loc
            max_scale=scale

    return maxi_loc,max_scale


while True:
    _, frame = cap.read()
    frame1=convert_to_binary(frame)
    contours, hierarchy = cv2.findContours(preprocess(frame1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)

        if 6 > sides > 4 and sides + 2 == len(approx) and len(approx) > 6:
            slopey1=(approx[1,0,1]-approx[2,0,1])/(approx[1,0,0]-approx[2,0,0])
            slopey2=(approx[3,0,1]-approx[4,0,1])/(approx[3,0,0]-approx[4,0,0])
            slopey3=(approx[5,0,1]-approx[6,0,1])/(approx[5,0,0]-approx[6,0,0])
            slopex1=(approx[2,0,1]-approx[3,0,1])/(approx[2,0,0]-approx[3,0,0])
            slopex2=(approx[4,0,1]-approx[5,0,1])/(approx[4,0,0]-approx[5,0,0])

            if((abs(slopey1-slopey2) < 0.5 or abs(slopey1-slopey3)<0.5 or abs(slopey2-slopey3)) and (abs(slopex1-slopex2)<0.5)):

                arrow_tip = find_tip(approx[:,0,:], hull.squeeze())
                if arrow_tip and len(approx)>=7:
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                    cv2.circle(frame, arrow_tip, 3, (0, 0, 255), cv2.FILLED)
                    if type(approx) != 'str':
                        dir=direction(approx)
                    else:
                        dir=-1
                    if dir!=-1: 
                        print(dir)


    match_loc,match_scale = multi_scale_template_matching(frame1, right)
    w=int(right.shape[1]*match_scale)
    h=int(left.shape[0]*match_scale)

    if match_loc!=-1:
        cv2.rectangle(frame, match_loc, (match_loc[0] + w, match_loc[1] + h), (0, 255, 0), )
        degree=math.degrees(math.atan((match_loc[0] - 320.0 + 0.5 * w)/match_loc[1]))
        direction='right'
        print(direction)

    match_loc,match_scale = multi_scale_template_matching(frame1, left)
    w=int(right.shape[1]*match_scale)
    h=int(left.shape[0]*match_scale)
    if match_loc!=-1:
        cv2.rectangle(frame, match_loc, (match_loc[0] + w, match_loc[1] + h), (255, 0, 0), 2)

    cv2.imshow("Image", frame)
    cv2.imshow("Image1", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
