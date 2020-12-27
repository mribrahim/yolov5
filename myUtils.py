import numpy as np


def area(box):
    return np.abs( (box[2]-box[0]) * (box[3]-box[1]) )

def findBestIndex(box, candidates):
    
    index, a = box[0], box[1]
    conf = a[1]
    maxArea = area([a[3][0], a[3][1], a[4][0], a[4][1]])
    for c in candidates:
        i = c[0]
        box = c[1]
        clss = box[0]
        if 1==clss or 2==clss:
            rect2 = [box[3][0], box[3][1], box[4][0], box[4][1]] 
            if area(rect2)>maxArea:
                maxArea = area(rect2)
                index = i
        elif box[1] > conf:
            conf = box[1]
            index = i

    return index


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	smallBoxArea = min(boxAArea, boxBArea)
	if interArea > 0.8 * smallBoxArea:
		return 1.0
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def delete_overlappings(bboxes, intersect_threshold):

    i = 0
    while i < len(bboxes):
        box = bboxes[i]
        a = [box[3][0], box[3][1], box[4][0], box[4][1]] 
        
        hasIntersect = False
        candidates = []
        for j in range(i, len(bboxes)):
            if i == j :
                continue

            boxTarget = bboxes[j]
            # if box[0] != boxTarget[0]: # check only same class
            #     continue

            b = [boxTarget[3][0], boxTarget[3][1], boxTarget[4][0], boxTarget[4][1]] 
            intersection = bb_intersection_over_union(a, b)
            if intersection > intersect_threshold:
                hasIntersect = True
                candidates.append([j, boxTarget])
        
        if hasIntersect:
            biggest = findBestIndex([i, box], candidates)
            bboxes[i] = bboxes[biggest]

            del_list = []
            for k in candidates:
                del_list.append(k[0])

            bboxes = [ bboxes[k] for k in range(len(bboxes)) if k not in del_list ]
        
        i += 1
    return bboxes