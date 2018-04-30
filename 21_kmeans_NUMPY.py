''' Compared to #20, it:
--reads the data points from a file points.txt
--has an optimized function D_sqr, with Numpy arrays
'''
import random
import sys
import numpy as np
from matplotlib import pyplot as pp

'''Don't touch this function yet!'''
def D_sqr(cent, poin, clos):
    k = len(cent)
    n = len(poin)
    s = 0.0
    for i in range(k):  #for each center
        for j in range(n):
            if i == clos[j]:
                s += (cent[i]-poin[j])**2
    return s   
    
def my_data_load(filename):
    with open(filename, 'r') as f:
        li = f.readlines()
    if len(li) == 0:
        print 'Empty file!!!!'
        sys.exit(0)
    a = np.array(map(float, li[0].split(',')), dtype='float32')
    for i in range(1, len(li)):
        temp = li[i].split(',')
        try:
            b = np.array(map(float, temp))
            a = np.hstack((a, b))
        except ValueError:
            print 'Conversion error!'
    return a

arr_points  = my_data_load('points.txt')
arr_closest = np.zeros_like(arr_points, dtype='int8')
n = arr_points.shape[0]     #nr. of points
print '\n', n
k = 3               #nr. of clusters


random.seed(42)     #for repeatability
centroids = []
for i in range(k):  
    centroids.append(random.uniform(arr_points[0], arr_points[-1]))
print 'Initial centroids:', centroids

new_centroids = [0]*k
change = 1.0

while change > 1e-4:    #main loop of K-means iteration
    #for each point, find which centroid is closest
    for i in range(n):
        distances = []
        for j in range(k):
            distances.append(abs(arr_points[i]-centroids[j]))
        arr_closest[i] = distances.index(min(distances))    #index where min occurs
    #find new centroids for each of k sets of points
    summation  = [0]*k
    counter    = [0]*k
    difference = [0]*k
    for i in range(n):
        summation[arr_closest[i]] += arr_points[i]
        counter[arr_closest[i]] += 1
    for j in range(k):
        if counter[j] == 0:
            print 'ERROR - Empty set!'
            sys.exit(0)
        new_centroids[j] = summation[j]/float(counter[j])
        difference[j] = abs(new_centroids[j] - centroids[j])
    change = sum(difference)
    #print change
    #print '    New centroids:', centroids
    centroids = new_centroids[:]
#end of K-means iterations
print '  Final centroids:', centroids    

#for each point, find which of the  final centroids is closest
for i in range(n):
    distances = []
    for j in range(k):
        distances.append(abs(arr_points[i]-centroids[j]))
    arr_closest[i] = distances.index(min(distances))    #index where min occurs
#print ' Closest centroid:', closest
#print 'Distortion:', D_sqr(centroids, points, closest)

color_dict = {0:'red', 1:'green', 2:'blue', 3:'yellow', 4:'brown'}
colors = [color_dict[x] for x in arr_closest] 

pp.figure(figsize=(10, 1))   
pp.scatter(arr_points   , np.zeros_like(arr_points),
           c=colors, s=200, alpha=0.7)
pp.scatter(arr_points   , np.zeros_like(arr_points),
           c=colors, s=20, alpha=1)
pp.scatter(centroids, [0]*len(centroids), c=color_dict.values(),
           s=1000, marker='|')
pp.ylim(-1, 1)
pp.xlim(0-10, arr_points[n-1]+10)

for i in range(k):
    pp.annotate(str(centroids[i]), xy=(centroids[i], 0.7),
                xytext = (centroids[i]+50, 1.3),
                arrowprops=dict(facecolor='black', width = 1,
                                headwidth = 6, headlength = 5))
pp.show()



