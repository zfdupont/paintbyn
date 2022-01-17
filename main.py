
import cv2
from math import sqrt
import numpy as np

test_image_path = './test/lenna.png'

image = cv2.imread(test_image_path)

def median_cut(img, steps=3):
    rows, cols, channels = img.shape
    buckets = [[]]
    for r in range(rows):
        for c in range(cols):
            px = [img.item(r,c,i) for i in range(3)]
            buckets[0].append(px)
    for step in range(steps):
        n = len(buckets)
        for _ in range(n):
            bucket = buckets.pop(0)
            b = max(px[0] for px in bucket) - min(px[0] for px in bucket)
            g = max(px[1] for px in bucket) - min(px[1] for px in bucket)
            r = max(px[2] for px in bucket) - min(px[2] for px in bucket)
            if b == max(b,g,r):
                bucket.sort(key=lambda x:x[0])
            elif g == max(b,g,r):
                bucket.sort(key=lambda x:x[1])
            else:
                bucket.sort(key=lambda x:x[2])
            median_index = len(bucket)//2
            first_half = bucket[:median_index]
            second_half = bucket[median_index:]
            buckets.extend([first_half, second_half])
    raw = [tuple(map(np.mean, zip(*bucket))) for bucket in buckets] # this has ugly decimals :(
    return list(tuple(map(int,tup)) for tup in raw)
            
def quantize(img):
    def closest(pixel, color_set):
        # simple 3-dim euclidean distance formula
        distance = lambda A,B: sqrt(sum((A[i]-B[i])**2 for i in range(3)))
        return min(color_set, key=lambda x: distance(x, pixel))
    reduced_set = median_cut(img, 4) # get color set from median cut algorithm
    
    rows, cols, channels = img.shape
    for r in range(rows):
        for c in range(cols):
            px = [img.item(r,c,i) for i in range(3)]
            new_px = closest(px, reduced_set)
            for i in range(3):
                img.itemset((r,c,i), new_px[i])


quantize(image)

cv2.imwrite('./test/lenna_reduced.png', image)