import cv2
import os


# path = 'dataset/training_set/'

# for r,d,f in os.walk(path):
#     for file in f:
#         if '.jpg' in file:
#             files.append(os.path.join(r,file))


# for f in files:
#     print(f)
test_set_path = 'dataset/training_set/Fucon'
new_test_path = 'dataset/new_training_set'
count = 0
for r in os.listdir(test_set_path):
    original_img =cv2.imread(test_set_path+'/'+r,1)
    newimg = cv2.resize(original_img,(48,48))
    path = os.path.join(new_test_path+'/Fucon',r)
    status =cv2.imwrite(path,newimg)
    count+=1
    print(path+':'+str(status)+':'+str(count))



    # for img in os.listdir(test_set_path+r+'/'):
    #         count+=1
    #         original_img =cv2.imread(test_set_path+r+'/'+img,1)
    #         newimg = cv2.resize(original_img,(48,48))
    #         path = os.path.join(new_test_path+'/'+r,img)
    #         #print(path)
            



#    for img in os.listdir(test_set_path+r+'/'):
#        count+=1
#        original_img =cv2.imread(test_set_path+r+'/'+img,1)
#        newimg = cv2.resize(original_img,(48,48))
#        path = os.path.join(new_test_path+'/'+r,img)
#        status =cv2.imwrite(path,newimg)
#       print(path+':'+str(status)+':'+str(count))

