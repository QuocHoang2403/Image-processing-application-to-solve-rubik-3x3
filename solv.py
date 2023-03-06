import cv2
import numpy as np
# from solving import *
from rubik_solver import utils 
label_check=True #hien thu tu sap xep contour
#Set color range for 6 colors
colors = {
    'yellow': ([32,53,65],[53,255,255]),#([31,0,65],[35,245,255]),
    'blue':   ([98,166,178],[125,255,255]),
    'green':  ([21,196,103],[96,255,255]),#([51,0,186],[81,255,255]),
    'red':    ([165,233,117],[179,255,241]),#([174,213,95],[179,255,255]),
    'white':  ([70,0,215],[116,219,255]),
    'orange': ([123,0,213],[179,255,255]),#([0,123,107],[10,255,255])
    }

 #######################################
def find_color(h,s,v):#finds color given h,s,v values(average)
    if ((123<=h<=179) and (213<=v<=255) and (0<=s<=255)): 
        return "o"
    elif (32<=h<=53) and (53<=s<=255) and (65<=v<=255):
        return "y"
    elif (21<=h<=96) and(196<=s<=255)  and (103<=v<=255): #h:81
        return "g"
    elif (70<h<=116) and (0<=s<=219) and (215<v<255):
        return "w"
    elif (98<=h<=125) and (178<=v<=255) and (166<=s<=255):
        return "b" 
    else:
        return 'r'

#Sort contours from 1-->9 
def sort_contours(cnts, method="top-to-bottom"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    try:
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]

        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
    except:
        print('wait to have contour')
         # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes

#Find contour of rubik
def find_contour(gray,label_check=label_check):
    global centers,rubik_str,hsv_avg,temp,final_string,points
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cnts,_)=sort_contours(contours, method="top-to-bottom")
    # Take each row of 3 and sort from left-to-right or right-to-left
    cube_rows = []
    row = []
    for (i, c) in enumerate(cnts, 1):
        row.append(c)
        if i % 3 == 0:  
            (cnts, _) = sort_contours(row, method="left-to-right")
            cube_rows.append(cnts)
            row = []

    # Draw text
    number = 0
    centers=[]
    for row in cube_rows:
        for c in row:
            x,y,w,h = cv2.boundingRect(c)
            (m,n),radius=cv2.minEnclosingCircle(c)
            #Average bgr in rois
            b_avg,g_avg,r_avg,_= np.array(cv2.mean(frame[y:y+h,x:x+w])).astype(int)
            # print(b_avg,g_avg,r_avg)
            hsv_avg=cv2.cvtColor(np.uint8([[[b_avg,g_avg,r_avg]]]),cv2.COLOR_BGR2HSV)
            list_color=hsv_avg.squeeze().tolist()
            h_avg= list_color[0]
            s_avg= list_color[1]
            v_avg= list_color[2]
            color=find_color(h_avg,s_avg,v_avg)
            list_color.append(color)
            list_color.append(int(m))
            list_color.append(int(n))
            centers.append(list_color)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 2)
            if label_check:
                cv2.putText(frame, "{}".format(color), (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                # cv2.putText(frame, "#{}".format(number + 1), (x+50,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            number += 1
              
def puttext(broke):
    if (broke==0):
        # Using cv2.putText() method
        cv2.putText(frame, 'show top face', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 2, cv2.LINE_AA)
    elif (broke==1):
        cv2.putText(frame, 'show left face', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 2, cv2.LINE_AA)
    elif (broke==2):
        cv2.putText(frame, 'show front face', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 2, cv2.LINE_AA)
    elif (broke==3):
        cv2.putText(frame, 'show right face', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 2, cv2.LINE_AA)
    elif (broke==4):
        cv2.putText(frame, 'show back face', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 2, cv2.LINE_AA)
    elif (broke==5):
        cv2.putText(frame, 'show bottom face', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 2, cv2.LINE_AA)
        broke=5 

#Processing image      
def processing_frame(frame):
    # filter noise in image
    image=cv2.GaussianBlur(frame,(5,5),0)
    #Convert hsv color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #Create mask 
    mask = np.zeros(image.shape, dtype=np.uint8)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    for color, (lower, upper) in colors.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        color_mask = cv2.inRange(image, lower, upper)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=6)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        color_mask = cv2.merge([color_mask, color_mask, color_mask])
        # draw the detected squares onto a mask
        mask = cv2.bitwise_or(mask, color_mask)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return gray
def Morph(frame_morph):
    kernels = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    frame_morph = cv2.morphologyEx(frame_morph,cv2.MORPH_ERODE,kernels,5)
    dist_img = cv2.distanceTransform(frame_morph,cv2.DIST_L2,3)
    cv2.normalize(dist_img,dist_img,0,1,cv2.NORM_MINMAX)
    frame_morph = cv2.threshold(dist_img,0.15,255,cv2.THRESH_BINARY)[1]
    frame_morph = np.uint8(frame_morph)
    frame_morph = cv2.morphologyEx(frame_morph,cv2.MORPH_DILATE,kernels,50)
    return frame_morph
def thresholdColor(frame_thresh):
    global center_points
    new_frame = np.zeros((frame_thresh.shape[0],frame_thresh.shape[1],1),dtype=np.uint8)

    ##Red
    R = cv2.inRange(frame_thresh, (139, 190, 45), (179, 255, 255))
    R = Morph(R)

    ##White
    W = cv2.inRange(frame_thresh, (61, 0, 162), (118, 255, 255))
    W = Morph(W)

    #Blue
    # B = cv2.inRange(frame_thresh, (60, 0, 130), (133, 255, 255))
    # B = Morph(B)

    ##Orange
    # O = cv2.inRange(frame_thresh, (3, 220, 173), (25, 248, 253))
    # O = Morph(O)

    ##yellow
    Y = cv2.inRange(frame_thresh, (31, 57, 57), (87, 255, 255))
    Y = Morph(Y)

    ##green
    G = cv2.inRange(frame_thresh, (19, 32, 148), (91, 255, 255))
    G = Morph(G)

    img = cv2.add(new_frame,R)
    img = cv2.add(img,W)
    # img = cv2.add(img,B)
    # img = cv2.add(img,O)
    img = cv2.add(img,Y)
    img = cv2.add(img,G)

    frame_draw = np.zeros_like(frame_thresh)
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center_points=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>1000:
            (x,y),_ = cv2.minEnclosingCircle(cnt)
            x = int(x)
            y = int(y)
            empty_list=[x,y]
            center_points.append(empty_list)
            cv2.rectangle(frame_draw,(x,y),(x+3,y+3),(255,255,255),2)

    return frame_draw
def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr

def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(mag*4,  255)
    # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return hsv

def detect_Area(flow_area):
    valid = draw_hsv(flow_area)
    valid  = cv2.inRange(valid, (0, 0, 180), (255, 255, 255)) #60--->200

    kernels = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

    valid = cv2.morphologyEx(valid,cv2.MORPH_ERODE,kernels,5)
    dist_img = cv2.distanceTransform(valid,cv2.DIST_L2,3)
    cv2.normalize(dist_img,dist_img,0,1,cv2.NORM_MINMAX)
    valid = cv2.threshold(dist_img,0.10,255,cv2.THRESH_BINARY)[1]
    valid = np.uint8(valid)
    valid = cv2.morphologyEx(valid,cv2.MORPH_DILATE,kernels,20)

    #area need to calculate
    flag = False
    area = []
    contours,hierarchy = cv2.findContours(valid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        flag = True
        x,y,w,h = cv2.boundingRect(cnt)
        a = [x,y,x+w,y+h]
        area.append(a)
    # area = np.array(area)
    return valid,area,flag
def detect_Directory(flow_dir):
    hsv_dir = draw_hsv(flow_dir)
    hsv_dir = cv2.cvtColor(hsv_dir,cv2.COLOR_HSV2BGR)
    _,area,flag = detect_Area(flow_dir)
    mean_a = []
    avr = 0
    if flag == True:
        matrix_point = np.array(area)
        for i,a in enumerate(area):
            x = matrix_point[i,0]
            y = matrix_point[i,1]
            x2 = matrix_point[i,2]
            y2 = matrix_point[i,3]
            mean =  np.average(hsv_dir[y:y2,x:x2,0])
            cv2.rectangle(hsv_dir,(x,y),(x2,y2),(255,255,255),2)
            mean_a.append(mean)
        # mean_a = np.average(mean_a)
        mean_a=np.mean(mean_a)
        avr = mean_a
    return hsv_dir,avr,flag
def drawArrow(p1,p2,centers_draw,frame_draw):
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    for i, ct in enumerate(centers_draw):
        if i == p1:
            x1 = ct[4]
            y1 = ct[5]
        elif i == p2:
            x2 = ct[4]
            y2 = ct[5]
    point1=(x1,y1)
    point2=(x2,y2)    
    cv2.arrowedLine(frame_draw,point1,point2,(255, 0, 255),2)
    # return frame_draw

# sap xep thu tu
def U(cube):
    cube_temp = cube.copy()
    cube_temp[0]=cube[6]
    cube_temp[1]=cube[3]
    cube_temp[2]=cube[0]
    cube_temp[3]=cube[7]
    cube_temp[5]=cube[1]
    cube_temp[8]=cube[2]
    cube_temp[9]=cube[18]
    cube_temp[10]=cube[19]
    cube_temp[11]=cube[20]
    cube_temp[18]=cube[27]
    cube_temp[19]=cube[28]
    cube_temp[20]=cube[29]
    cube_temp[27]=cube[36]
    cube_temp[28]=cube[37]
    cube_temp[29]=cube[38]
    cube_temp[36]=cube[9]
    cube_temp[37]=cube[10]
    cube_temp[38]=cube[11]
    cube_temp[6]=cube[8]
    cube_temp[7]=cube[5]
    return cube_temp
def U_CCW(cube):
    cube_temp = cube.copy()
    cube_temp[0]=cube[2]
    cube_temp[1]=cube[5]
    cube_temp[2]=cube[8]
    cube_temp[3]=cube[1]
    cube_temp[5]=cube[7]
    cube_temp[6]=cube[0]
    cube_temp[7]=cube[3]
    cube_temp[8]=cube[6]
    cube_temp[9]=cube[36]
    cube_temp[10]=cube[37]
    cube_temp[11]=cube[38]
    cube_temp[18]=cube[9]
    cube_temp[19]=cube[10]
    cube_temp[20]=cube[11]
    cube_temp[27]=cube[18]
    cube_temp[28]=cube[19]
    cube_temp[29]=cube[20]
    cube_temp[36]=cube[27]
    cube_temp[37]=cube[28]
    cube_temp[38]=cube[29]
    return cube_temp
def F(cube):
    cube_temp = cube.copy()
    cube_temp[18]=cube[24]
    cube_temp[19]=cube[21]
    cube_temp[20]=cube[18]
    cube_temp[23]=cube[19]
    cube_temp[26]=cube[20]
    cube_temp[25]=cube[23]
    cube_temp[24]=cube[26]
    cube_temp[21]=cube[25]
    cube_temp[6]=cube[17]
    cube_temp[7]=cube[14]
    cube_temp[8]=cube[11]
    cube_temp[27]=cube[6]
    cube_temp[30]=cube[7]
    cube_temp[33]=cube[8]
    cube_temp[45]=cube[33]
    cube_temp[46]=cube[30]
    cube_temp[47]=cube[27]
    cube_temp[11]=cube[45]
    cube_temp[14]=cube[46]
    cube_temp[17]=cube[47]
    return cube_temp
def F_CCW(cube):
    cube_temp = cube.copy()
    cube_temp[18]=cube[20]
    cube_temp[19]=cube[23]
    cube_temp[20]=cube[26]
    cube_temp[23]=cube[25]
    cube_temp[26]=cube[24]
    cube_temp[25]=cube[21]
    cube_temp[24]=cube[18]
    cube_temp[21]=cube[19]
    cube_temp[6]=cube[27]
    cube_temp[7]=cube[30]
    cube_temp[8]=cube[33]
    cube_temp[27]=cube[47]
    cube_temp[30]=cube[46]
    cube_temp[33]=cube[45]
    cube_temp[45]=cube[11]
    cube_temp[46]=cube[14]
    cube_temp[47]=cube[17]
    cube_temp[11]=cube[8]
    cube_temp[14]=cube[7]
    cube_temp[17]=cube[6]
    return cube_temp
def L(cube):
    cube_temp = cube.copy()
    cube_temp[9]=cube[15]
    cube_temp[10]=cube[12]
    cube_temp[11]=cube[9]
    cube_temp[14]=cube[10]
    cube_temp[17]=cube[11]
    cube_temp[16]=cube[14]
    cube_temp[15]=cube[17]
    cube_temp[12]=cube[16]
    cube_temp[0]=cube[44]
    cube_temp[3]=cube[41]
    cube_temp[6]=cube[38]
    cube_temp[18]=cube[0]
    cube_temp[21]=cube[3]
    cube_temp[24]=cube[6]
    cube_temp[45]=cube[18]
    cube_temp[48]=cube[21]
    cube_temp[51]=cube[24]
    cube_temp[38]=cube[51]
    cube_temp[41]=cube[48]
    cube_temp[44]=cube[45]
    return cube_temp
def L_CCW(cube):
    cube_temp = cube.copy()
    cube_temp[9]=cube[11]
    cube_temp[10]=cube[14]
    cube_temp[11]=cube[17]
    cube_temp[14]=cube[16]
    cube_temp[17]=cube[15]
    cube_temp[16]=cube[12]
    cube_temp[15]=cube[9]
    cube_temp[12]=cube[10]
    cube_temp[0]=cube[18]
    cube_temp[3]=cube[21]
    cube_temp[6]=cube[24]
    cube_temp[18]=cube[45]
    cube_temp[21]=cube[48]
    cube_temp[24]=cube[51]
    cube_temp[45]=cube[44]
    cube_temp[48]=cube[41]
    cube_temp[51]=cube[38]
    cube_temp[38]=cube[6]
    cube_temp[41]=cube[3]
    cube_temp[44]=cube[0]
    return cube_temp
def R(cube):
    cube_temp = cube.copy()
    cube_temp[27]=cube[33]
    cube_temp[28]=cube[30]
    cube_temp[29]=cube[27]
    cube_temp[32]=cube[28]
    cube_temp[35]=cube[29]
    cube_temp[34]=cube[32]
    cube_temp[33]=cube[35]
    cube_temp[30]=cube[34]
    cube_temp[2]=cube[20]
    cube_temp[5]=cube[23]
    cube_temp[8]=cube[26]
    cube_temp[20]=cube[47]
    cube_temp[23]=cube[50]
    cube_temp[26]=cube[53]
    cube_temp[36]=cube[8]
    cube_temp[39]=cube[5]
    cube_temp[42]=cube[2]
    cube_temp[47]=cube[42]
    cube_temp[50]=cube[39]
    cube_temp[53]=cube[36]
    return cube_temp
def R_CCW(cube):
    cube_temp = cube.copy()
    cube_temp[27]=cube[29]
    cube_temp[28]=cube[32]
    cube_temp[29]=cube[35]
    cube_temp[32]=cube[34]
    cube_temp[35]=cube[33]
    cube_temp[34]=cube[30]
    cube_temp[33]=cube[27]
    cube_temp[30]=cube[28]
    cube_temp[2]=cube[42]
    cube_temp[5]=cube[39]
    cube_temp[8]=cube[36]
    cube_temp[20]=cube[2]
    cube_temp[23]=cube[5]
    cube_temp[26]=cube[8]
    cube_temp[36]=cube[53]
    cube_temp[39]=cube[50]
    cube_temp[42]=cube[47]
    cube_temp[47]=cube[20]
    cube_temp[50]=cube[23]
    cube_temp[53]=cube[26]
    return cube_temp
def D(cube):
    cube_temp = cube.copy()
    cube_temp[45]=cube[51]
    cube_temp[46]=cube[48]
    cube_temp[47]=cube[45]
    cube_temp[50]=cube[46]
    cube_temp[53]=cube[47]
    cube_temp[52]=cube[50]
    cube_temp[51]=cube[53]
    cube_temp[48]=cube[52]
    cube_temp[15]=cube[42]
    cube_temp[16]=cube[43]
    cube_temp[17]=cube[44]
    cube_temp[24]=cube[15]
    cube_temp[25]=cube[16]
    cube_temp[26]=cube[17]
    cube_temp[33]=cube[24]
    cube_temp[34]=cube[25]
    cube_temp[35]=cube[26]
    cube_temp[42]=cube[33]
    cube_temp[43]=cube[34]
    cube_temp[44]=cube[35]
    return cube_temp
def D_CCW(cube):
    cube_temp = cube.copy()
    cube_temp[45]=cube[47]
    cube_temp[46]=cube[50]
    cube_temp[47]=cube[53]
    cube_temp[50]=cube[52]
    cube_temp[53]=cube[51]
    cube_temp[52]=cube[48]
    cube_temp[51]=cube[45]
    cube_temp[48]=cube[46]
    cube_temp[15]=cube[24]
    cube_temp[16]=cube[25]
    cube_temp[17]=cube[26]
    cube_temp[24]=cube[33]
    cube_temp[25]=cube[34]
    cube_temp[26]=cube[35]
    cube_temp[33]=cube[42]
    cube_temp[34]=cube[43]
    cube_temp[35]=cube[44]
    cube_temp[42]=cube[15]
    cube_temp[43]=cube[16]
    cube_temp[44]=cube[17]
    return cube_temp
def B(cube):
    cube_temp = cube.copy()
    cube_temp[36]=cube[42]
    cube_temp[37]=cube[39]
    cube_temp[38]=cube[36]
    cube_temp[41]=cube[37]
    cube_temp[44]=cube[38]
    cube_temp[43]=cube[41]
    cube_temp[42]=cube[44]
    cube_temp[39]=cube[43]
    cube_temp[0]=cube[29]
    cube_temp[1]=cube[32]
    cube_temp[2]=cube[35]
    cube_temp[9]=cube[2]
    cube_temp[12]=cube[1]
    cube_temp[15]=cube[0]
    cube_temp[51]=cube[9]
    cube_temp[52]=cube[12]
    cube_temp[53]=cube[15]
    cube_temp[29]=cube[53]
    cube_temp[32]=cube[52]
    cube_temp[35]=cube[51]
    return cube_temp
def B_CCW(cube):
    cube_temp = cube.copy()
    cube_temp[36]=cube[38]
    cube_temp[37]=cube[41]
    cube_temp[38]=cube[44]
    cube_temp[41]=cube[43]
    cube_temp[44]=cube[42]
    cube_temp[43]=cube[39]
    cube_temp[42]=cube[36]
    cube_temp[39]=cube[37]
    cube_temp[0]=cube[15]
    cube_temp[1]=cube[12]
    cube_temp[2]=cube[9]
    cube_temp[9]=cube[51]
    cube_temp[12]=cube[52]
    cube_temp[15]=cube[53]
    cube_temp[51]=cube[35]
    cube_temp[52]=cube[32]
    cube_temp[53]=cube[29]
    cube_temp[29]=cube[0]
    cube_temp[32]=cube[1]
    cube_temp[35]=cube[2]
    return cube_temp
    
def check(predict,rubik_str):
    check_val=0
    if rubik_str ==  predict[18:27]:
        check_val=1
    return check_val
def checkBF(predict,rubik_str):
    check_val=0
    if rubik_str ==  predict[27:36]:
        check_val=1
    return check_val

def check_step(solver,t,step):
    if(len(step)>1):
        if(step=="F2"):
            solver.remove(step)
            solver.insert(id,"right")
            solver.insert(id,"L")
            solver.insert(id,"L")
            solver.insert(id,"left")
            
        elif(step=="B2"):
            solver.remove(step)
            solver.insert(id,"right")
            solver.insert(id,"R")
            solver.insert(id,"R")
            solver.insert(id,"left")
            
        elif(step[1]=='2'):
            solver.remove(step)
            solver.insert(id,t)
            solver.insert(id,t)
            
        elif(step=="B'"):
            solver.remove(step)
            solver.insert(id,'right')
            solver.insert(id,"R'")
            solver.insert(id,"left")
            
        elif(step=="F'"):
            solver.remove(step)
            solver.insert(id,'right')
            solver.insert(id,"L'")
            solver.insert(id,"left")
        return solver
            
    else:
        if(step=="B"):
            solver.remove(step)
            solver.insert(id,'right')
            solver.insert(id,"R")
            solver.insert(id,"left")
            
        elif(step=="F"):
            solver.remove(step)
            solver.insert(id,"right")
            solver.insert(id,"L")
            solver.insert(id,"left")
        return solver

def solve_rubik(final_string):
    solver=utils.solve(final_string,"Kociemba")
    for id,step in enumerate(solver):
        step=str(step) # B2 R L' L2
        t=str(step[0]) # R R F L r B R
        solver=check_step(solver,t,step)
    flag=2
    return solver,flag
def check_string(centers):
    rubik_str=""
    if(len(centers)==9):
        for i in range(9):
            rubik_str=rubik_str+str(centers[i][3])
    rubik_str=[str(i) for i in rubik_str]
    return rubik_str    
def solve_single_step(count,final_string):
    global check_BF
    if solver[count] == "U":
        drawArrow(2,0,centers,frame2)
        predict = U(final_string)
        check_val = check(predict,rubik_str)
        if check_val ==1:
            count+=1
            check_val=0
            final_string = predict
    elif solver[count] =="U'":
        drawArrow(0,2,centers,frame2)
        predict = U_CCW(final_string)
        check_val = check(predict,rubik_str)
        if check_val ==1:
            count+=1
            check_val=0
            final_string = predict
    elif solver[count]  =="D":
        drawArrow(6,8,centers,frame2)
        predict = D(final_string)
        check_val = check(predict,rubik_str)
        if check_val ==1:
            count+=1
            check_val=0
            final_string = predict
    elif solver[count]  =="D'":
        drawArrow(8,6,centers,frame2)
        predict = D_CCW(final_string)
        check_val = check(predict,rubik_str)
        if check_val ==1:
            count+=1
            check_val=0
            final_string = predict
    elif solver[count]  =="L":
        drawArrow(0,6,centers,frame2)
        if check_BF ==1:
            predict = F(final_string)
            check_val = checkBF(predict,rubik_str)
            print(check_val)
        else:
            predict = L(final_string)
            check_val = check(predict,rubik_str)
        if check_val ==1:
            count+=1
            check_val=0
            final_string = predict
    elif solver[count]  =="L'":
        drawArrow(6,0,centers,frame2)
        if check_BF ==1:
            predict = F_CCW(final_string)
            check_val = checkBF(predict,rubik_str)
            print(check_val)
        else:
            predict = L_CCW(final_string)
            check_val = check(predict,rubik_str)
        if check_val ==1:
            count+=1
            check_val=0
            final_string = predict
    elif solver[count]  =="R":
        drawArrow(8,2,centers,frame2)
        if check_BF ==1:
            predict = B(final_string)
            check_val = checkBF(predict,rubik_str)
            print(predict[27:36])
            print(rubik_str)
            # print(check_val)
        else:
            predict = R(final_string)
            check_val = check(predict,rubik_str)
        if check_val ==1:
            count+=1
            check_val=0
            final_string = predict

    elif solver[count]  =="R'":
        drawArrow(2,8,centers,frame2)
        if check_BF ==1:
            predict = B_CCW(final_string)
            check_val = checkBF(predict,rubik_str)
        else:
            predict = R_CCW(final_string)
            check_val = check(predict,rubik_str)
        if check_val ==1:
            count+=1
            check_val=0
            final_string = predict
    elif solver[count]  =="right":
        drawArrow(0,2,centers,frame2)
        drawArrow(3,5,centers,frame2)
        drawArrow(6,8,centers,frame2)
        if final_string[18:27] == rubik_str:
            count+=1
            check_BF=0
    elif solver[count]  =="left":
        drawArrow(2,0,centers,frame2)
        drawArrow(5,3,centers,frame2)
        drawArrow(8,6,centers,frame2)
        if final_string[27:36] == rubik_str:
            count+=1
            check_BF=1
def string_capture():
    broke=broke+1
    for i in range(9):
        rubik_str=rubik_str+str(centers[i][3])      
    if(len(rubik_str)==9):
        temp2=rubik_str
        if(len(final_string)<54):
            if(temp2!=temp):
                final_string+=temp2
                temp=temp2 
                print(final_string)
                print(len(final_string))
        rubik_str =""
        temp2='' 
def detect(broke,final_string,rubik_str):
    while True:
        ret, frame = vid.read()
        # Display the resulting frame
        puttext(broke)
        #Image processing
        gray=processing_frame(frame)
        find_contour(gray,label_check=label_check)
        if(len(centers)==9):
            if k==ord('s'):
                print("Image Captured")
                string_capture()

            if(len(final_string)==54):
                flag=1
                # print('hello')
                break                         
        cv2.imshow('frame1',gray)
        cv2.imshow('frame',frame)
        k=cv2.waitKey(1) & 0xff
        if k==27:#ESC is pressed
            break

# define a video capture object
vid = cv2.VideoCapture(0)   
vid.set(cv2.CAP_PROP_BRIGHTNESS, 1)  
vid.set(cv2.CAP_PROP_SATURATION,195 )
vid.set(cv2.CAP_PROP_CONTRAST,190)
vid.set(cv2.CAP_PROP_FPS,60)
vid.set(cv2.CAP_PROP_TEMPERATURE,4500)
ret,frame1 = vid.read()
frame_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
prevgray = thresholdColor(frame_hsv)
prevgray = cv2.cvtColor(prevgray, cv2.COLOR_BGR2GRAY)
broke=0
centers=[]
center_points=[]
temp='huy'
temp2='ghy'
rubik_str=''
i=0
points=[]
temp='huy'
rubik_str=''
id=0
flag = 0
count =-1
flag_step =0
check_BF =0
final_string=''
predict=''
check_val=0
DEBUG=False
# Just test solving step
if DEBUG:
    flag = 1
    final_string='gyygyyrggooyybyooobrrbrrybgwrrwgwyrrggwooowgwbwobwbbwb'
while(True): 
    # Capture the video frame by frame 
    ret, frame = vid.read()
    _,frame2 = vid.read()
    #Solving rubik string to get list of step
    if(flag==1):
        #Processing step and turn on rubik solving instruction mode (flag=2)
        solver,flag=solve_rubik(final_string)

    #Solve rubik and draw arrow (rubik solving instruction mode)
    elif(flag==2 ):
        #Image processing
        gray=processing_frame(frame2)
        #Find contour
        find_contour(gray,label_check=label_check)
        #check whether 9 contours are detected or not,if detected,add to final string
        rubik_str=check_string(centers)
        final_string=[str(i) for i in final_string]
        #Set count value in range (0,length of solving step)
        if count >= 0 and count < len(solver):
            solve_single_step(count,final_string) 
        #Press c button to turn on solving step instruction
        if k==ord('c'): 
            count +=1
        # all steps are solved,show text 'done' on screen
        if(count>=len(solver)):
            cv2.putText(frame2, "DONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame',frame2)
    cv2.imshow('frame_main',frame)
    k=cv2.waitKey(1) & 0xff
    if k==27:#ESC is pressed, break loop and stop frame
        break
    #Delete last 9 strings character if detect wrong
    if k==ord('d'):
        final_string=final_string[:-9]
    #SPACE is pressed then turn on detection mode
    elif k==32:
        if(flag==0):
            #Detect contours,get strings and turn on solving rubik mode(flag=1)
            detect(broke,final_string,rubik_str)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
