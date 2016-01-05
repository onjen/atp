'''
class image

defines the image path, the keypoints, the descriptors
and the associated brick
'''
class Image:
    def __init__(self, keypoints, descriptors, filename, brick_id):
        self.keypoints_array = pickle_keypoints(keypoints, descriptors)
        self.filename = filename
        self.brick_id = brick_id

# format keypoints to use pickle to serialize keypoints and  descriptors
def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])
        i += 1
        temp_array.append(temp)
    return temp_array

