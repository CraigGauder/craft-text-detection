import os, sys
import numpy as np
import cv2
import pickle
import scipy.io
import scipy.stats as stats
from PIL import Image
from shapely import geometry as geo


class SynthTextDataPreprocesser:
    def __init__(self):
        def generate_base_gaussian(base_dim):
            x, y = np.meshgrid(range(base_dim), range(base_dim), indexing='ij')
            d = np.sqrt(np.abs(base_dim/2 - x)**2 + np.abs(base_dim/2 - y)**2)
            sigma, mu = base_dim/4, 0.0
            g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
            return g
        self.base_gaussian = generate_base_gaussian(base_dim=512)
        
        self.y_target_maps = None
        self.x_input_images = None
        
        
    def createRegionScoreMap(self, image, charBB):
        '''
        image - (3 x H x W)
        charBB - (4 x 2 x N_chars)
        '''
        base_dim = self.base_gaussian.shape[0]
        base_gaussian_corners = np.array([(0,0), (0,base_dim), (base_dim, base_dim), (base_dim, 0)])

        warpedGaussians = []
        for i in range(charBB.shape[-1]):
            h, mask = cv2.findHomography(base_gaussian_corners, charBB[:, :, i]//2, cv2.RANSAC,5.0)
            try:
                warpedGaussian = cv2.warpPerspective(self.base_gaussian, h, (image.shape[0]//2, image.shape[1]//2))
            except Exception as e:
                print(f'error finding homography: {e}')
                return None
            warpedGaussians.append(warpedGaussian)

        return sum(warpedGaussians)
    
    @staticmethod
    def getAffinityBox(char1, char2):
        '''
        char1/char2 - (4 x 2 (x 1))
        '''
        char1 = char1.squeeze()
        char2 = char2.squeeze()
        assert(char1.shape == (4,2))
        assert(char2.shape == (4,2))

        char1_points = [
            (p[0], p[1]) for p in char1
        ]
        char2_points = [
            (p[0], p[1]) for p in char2
        ]

        char1_diag1 = geo.LineString([char1_points[0], char1_points[2]])
        char1_diag2 = geo.LineString([char1_points[1], char1_points[3]])
        char1_mid = char1_diag1.intersection(char1_diag2).coords[0]
        char1_out1 = geo.Polygon(char1_points[:2] + [char1_mid]).centroid.coords[0]
        char1_out2 = geo.Polygon(char1_points[2:] + [char1_mid]).centroid.coords[0]

        char2_diag1 = geo.LineString([char2_points[0], char2_points[2]])
        char2_diag2 = geo.LineString([char2_points[1], char2_points[3]])
        char2_mid = char1_diag1.intersection(char1_diag2).coords[0]
        char2_out1 = geo.Polygon(char2_points[:2] + [char2_mid]).centroid.coords[0]
        char2_out2 = geo.Polygon(char2_points[2:] + [char2_mid]).centroid.coords[0]
        
        return np.array([
            char1_out1,
            char2_out1,
            char2_out2,
            char1_out2
        ])
    
    @staticmethod
    def getCleanWordsFromText(txt):
        return np.array([w for x in txt for w in x.strip().split()])
    
    def createAffinityScoreMap(self, image, charBB, text):
        '''
        image - (3 x H x W)
        charBB - (4 x 2 x N_chars)
        text - (N_words,)
        '''
        base_dim = self.base_gaussian.shape[0]
        base_gaussian_corners = np.array([(0,0), (0,base_dim), (base_dim, base_dim), (base_dim, 0)])
        affinity_gaussians = []
        curr_char_pos = 0
        for w in self.getCleanWordsFromText(text):
            for i in range(curr_char_pos, curr_char_pos+len(w)-1):
                affinity_box = self.getAffinityBox(charBB[:, :, i], charBB[:, :, i+1])
                h, mask = cv2.findHomography(base_gaussian_corners, affinity_box//2, cv2.RANSAC,5.0)
                try:
                    warpedGaussian = cv2.warpPerspective(self.base_gaussian, h, (image.shape[0]//2, image.shape[1]//2))
                except Exception as e:
                    print(f'error finding homography: {e}')
                    return None
                affinity_gaussians.append(warpedGaussian)
            curr_char_pos += len(w)
        
        return sum(affinity_gaussians)
    
    def getImageAndLabelData(self, data_dir):
        '''
        data_dir: full path to a directory containing SynthText dataset 
        (i.e. structure batchNum/<images>)
        '''
        data_files = []
        data_labels = None
        for dir_name, dirs, files in os.walk(data_dir):
            print(f'reading data from files in {dir_name}')
            if 'gt.mat' in files:
                print('reading label data from gt.mat')
                data_labels = scipy.io.loadmat(os.path.join(DATA_DIR, "gt.mat"))
                imNames = np.array([x[0] for x in data_labels['imnames'][0]])
                charBB = data_labels['charBB'][0]
                wordBB = data_labels['wordBB'][0]
                txt = data_labels['txt'][0]
            if not dirs and all([x.lower().endswith('jpg') or x.lower().endswith('jpeg') for x in files]):
                data_files += [os.path.join(dir_name.replace(DATA_DIR, '').strip('/'), im) for im in files]

        if data_labels is None:
            raise Exception('gt.mat must be included in DATA_DIR')

        data_file_selector = np.isin(imNames, data_files)
        imNames = imNames[data_file_selector]
        charBB = charBB[data_file_selector]
        wordBB = wordBB[data_file_selector]
        txt = txt[data_file_selector]
        
        x_data = []
        y_data = []
        for i, imName in enumerate(imNames):
            im = np.asarray(Image.open(os.path.join(data_dir, imName)))
            char = charBB[i].transpose((1, 0, 2))
            regionScoreMap = self.createRegionScoreMap(im, char)
            affinityScoreMap = self.createAffinityScoreMap(im, char, txt[i])
            if regionScoreMap is not None and affinityScoreMap is not None:
                y_data.append(np.array([regionScoreMap, affinityScoreMap]))
                x_data.append(im)
            else:
                print(f'image {imName} removed from training set (could not find homography matrices for all chars for perspective transform)')
        assert (len(x_data) == len(y_data))
        print(f'Number of training examples/labels generated: {len(x_data)}')
        
        with open(os.path.join(DATA_DIR, 'dataImages.pickle'), 'wb') as file:
            pickle.dump(x_data, file)
        with open(os.path.join(DATA_DIR, 'dataLabels.pickle'), 'wb') as file:
            pickle.dump(y_data, file)
        print(f'saved x_data to {os.path.join(DATA_DIR, "dataImages.pickle")}')
        print(f'saved y_date to {os.path.join(DATA_DIR, "dataLabels.pickle")}')
        
        return x_data, y_data



if __name__ == '__main__':
    DATA_DIR = "/Users/craiggauder/ML/craft-text-detection/data/SynthText"

    print(f'processing data in {DATA_DIR}')
    data_processor = SynthTextDataPreprocesser()
    x, y = data_processor.getImageAndLabelData(DATA_DIR)

    assert(len(x) == len(y))
    print(f'dataImages.pickle contents -- complete bank of {len(x)} input images for model')
    print(f'dataLabels.pickle contents -- {len(y)} corresponding label heat maps for {len(x)} input images')

    with open(os.path.join(DATA_DIR, 'dataLabels.pickle'), 'rb') as file:
        dataLabels = pickle.load(file)

    print('dataLabel example (index 0):')
    print(dataLabels[0])




