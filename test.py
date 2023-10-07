import numpy as np
import cv2

p1 = np.array([120, 100], dtype=np.float32)
p2 = np.array([230, 80], dtype=np.float32)
p3 = np.array([246, 165], dtype=np.float32)
p4 = np.array([108, 179], dtype=np.float32)

def L2(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

if __name__ == '__main__':

    mat = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.line(mat, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (240, 120, 30), 2)
    cv2.line(mat, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (240, 120, 30), 2)
    cv2.line(mat, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), (240, 120, 30), 2)
    cv2.line(mat, (int(p4[0]), int(p4[1])), (int(p1[0]), int(p1[1])), (240, 120, 30), 2)

    # Clockwise-L2
    D_p12 = L2(p1, p2)
    D_p23 = L2(p2, p3)
    D_p34 = L2(p3, p4)
    D_p41 = L2(p4, p1)
    
    # Shared vertice with clockwise-L2
    D_p14 = L2(p1, p4)
    D_p21 = L2(p2, p1)
    D_p32 = L2(p3, p2)
    D_p43 = L2(p4, p3)
    
    r1 = min(D_p12, D_p14)
    r2 = min(D_p21, D_p23)
    r3 = min(D_p34, D_p32)
    r4 = min(D_p41, D_p43)
    
    print("r1, r2, r3, r4:", r1, r2, r3, r4) 
    
    # inward_p1 must increase x and increase y
    
    pair_D1234 = np.mean([D_p12, D_p34]) # TODO: can be simplified to just add.
    pair_D2341 = np.mean([D_p23, D_p41])
    
    p_array = np.array([p1, p2, p3, p4]) # from Top left to Bottom right clockwisely
    r_array = np.array([r1, r2, r3, r4])
    
    R = 0.3
    if pair_D1234 >= pair_D2341: # Shrink pair 12 and 34 first and then 2341
        # Shrink 12 34
        
        direction = (p2 - p1) # 1
        norm_direction = (direction)/np.linalg.norm(direction)
        p1 = p1 +  norm_direction * (R * r1)
        
        direction = (p1 - p2) # 2
        norm_direction = (direction)/np.linalg.norm(direction)
        p2 = p2 + norm_direction * (R * r2)
        
        direction = (p4 - p3) # 3
        norm_direction = (direction)/np.linalg.norm(direction)
        p3 = p3 +  norm_direction * (R * r3)
        
        direction = (p3 - p4) # 4
        norm_direction = (direction)/np.linalg.norm(direction)
        p4 = p4 + norm_direction * (R * r4)
        
        # Shrink 23 41
    
        direction = (p3 - p2) # 2
        norm_direction = (direction)/np.linalg.norm(direction)
        p2 = p2 +  norm_direction * (R * r2)
        
        direction = (p2 - p3) # 3
        norm_direction = (direction)/np.linalg.norm(direction)
        p3 = p3 + norm_direction * (R * r3)
        
        direction = (p1 - p4) # 4
        norm_direction = (direction)/np.linalg.norm(direction)
        p4 = p4 +  norm_direction * (R * r4)
        
        direction = (p4 - p1) # 1
        norm_direction = (direction)/np.linalg.norm(direction)
        p1 = p1 + norm_direction * (R * r1)
    
    else:
        # Shrink 23 41
    
        direction = (p3 - p2) # 2
        norm_direction = (direction)/np.linalg.norm(direction)
        p2 = p2 +  norm_direction * (R * r2)
        
        direction = (p2 - p3) # 3
        norm_direction = (direction)/np.linalg.norm(direction)
        p3 = p3 + norm_direction * (R * r3)
        
        direction = (p1 - p4) # 4
        norm_direction = (direction)/np.linalg.norm(direction)
        p4 = p4 +  norm_direction * (R * r4)
        
        direction = (p4 - p1) # 1
        norm_direction = (direction)/np.linalg.norm(direction)
        p1 = p1 + norm_direction * (R * r1)

        # Shrink 12 34
        
        direction = (p2 - p1) # 1
        norm_direction = (direction)/np.linalg.norm(direction)
        p1 = p1 +  norm_direction * (R * r1)
        
        direction = (p1 - p2) # 2
        norm_direction = (direction)/np.linalg.norm(direction)
        p2 = p2 + norm_direction * (R * r2)
        
        direction = (p4 - p3) # 3
        norm_direction = (direction)/np.linalg.norm(direction)
        p3 = p3 +  norm_direction * (R * r3)
        
        direction = (p3 - p4) # 4
        norm_direction = (direction)/np.linalg.norm(direction)
        p4 = p4 + norm_direction * (R * r4)
        
    cv2.line(mat, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (50, 255, 50), 2)
    cv2.line(mat, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (50, 255, 50), 2)
    cv2.line(mat, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), (50, 255, 50), 2)
    cv2.line(mat, (int(p4[0]), int(p4[1])), (int(p1[0]), int(p1[1])), (50, 255, 50), 2)

    p1 = [int(x) for x in p1]
    p2 = [int(x) for x in p2]
    p3 = [int(x) for x in p3]
    p4 = [int(x) for x in p4]
    
    poly = np.array([p1, p2, p3, p4])
    poly = poly.reshape((-1, 1, 2))
    
    cv2.fillPoly(mat, [poly], (255, 255, 255))
    
    cv2.imshow("kekw", mat)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

    
    