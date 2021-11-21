'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import pandas as pd

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix, self.train_num = self.load_rating_file_as_matrix(path + "/train.csv")
        self.testRatings, self.test_num = self.load_rating_file_as_matrix(path + "/test.csv")
        self.textualfeatures,self.imagefeatures, = self.load_features(path)
        self.num_users, self.num_items = self.trainMatrix.shape
        print(self.train_num+self.test_num)

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items, num_total = 0, 0, 0
        df = pd.read_csv(filename, index_col=None, usecols=None)
        for index, row in df.iterrows():
            u, i = int(row['userID']), int(row['itemID'])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        for index, row in df.iterrows():
            user, item ,rating = int(row['userID']), int(row['itemID']) ,1.0
            if (rating > 0):
                mat[user, item] = 1.0
                num_total += 1
        return mat, num_total

    def load_features(self,data_path):
        import os
        from gensim.models.doc2vec import Doc2Vec
        # Prepare textual feture data.
        #doc2vec_model = Doc2Vec.load(os.path.join(data_path, 'doc2vecFile'))
        doc2vec_model = np.load(os.path.join(data_path, 'review.npz'), allow_pickle=True)['arr_0'].item()
        #print(doc2vec_model)
        vis_vec = np.load(os.path.join(data_path, 'image_feature.npy'), allow_pickle=True).item()
        #print(vis_vec)
        filename = data_path + '/train.csv'
        filename_test =  data_path + '/test.csv'
        df = pd.read_csv(filename, index_col=None, usecols=None)
        df_test = pd.read_csv(filename_test, index_col=None, usecols=None)
        num_items = 0
        asin_i_dic = {}
        for index, row in df.iterrows():
            asin, i = row['asin'], int(row['itemID'])
            asin_i_dic[i] = asin
            num_items = max(num_items, i)
        for index, row in df_test.iterrows():
            asin, i = row['asin'], int(row['itemID'])
            asin_i_dic[i] = asin
            num_items = max(num_items, i)
        features = []
        image_features = []
        for i in range(num_items+1):
            #print('doc')
            features.append(doc2vec_model[asin_i_dic[i]][0])
            #print('image')
            image_features.append(vis_vec[asin_i_dic[i]])
        return np.asarray(features,dtype=np.float32),np.asarray(image_features,dtype=np.float32)
