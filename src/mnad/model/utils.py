import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data


rng = np.random.RandomState(2020)


def list_frame_files(video_path):
    frames = []
    for pattern in ('*.jpg', '*.png'):
        frames.extend(glob.glob(os.path.join(video_path, pattern)))
    return sorted(frames)

def np_load_frame(filename, resize_height, resize_width):
    """Load and resize a BGR image as a float32 array in [-1, 1]."""
    image_decoded = cv2.imread(filename)
    if image_decoded is None:
        raise RuntimeError('Cannot decode frame: {}'.format(filename))
    if image_decoded.shape[:2] == (resize_height, resize_width):
        image_resized = image_decoded
    else:
        image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()
        
        
    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = os.path.basename(video)
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = list_frame_files(video)
            self.videos[video_name]['frame_index'] = {
                frame_path: frame_index
                for frame_index, frame_path in enumerate(self.videos[video_name]['frame'])
            }
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            
            
    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = os.path.basename(video)
            for i in range(len(self.videos[video_name]['frame']) - self._time_step - self._num_pred + 1):
                frames.append(self.videos[video_name]['frame'][i])
                           
        return frames               
            
        
    def __getitem__(self, index):
        video_name = os.path.basename(os.path.dirname(self.samples[index]))
        frame_name = self.videos[video_name]['frame_index'][self.samples[index]]
        
        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0)
        
        
    def __len__(self):
        return len(self.samples)


