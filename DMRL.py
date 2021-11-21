import functools
import numpy as np
import tensorflow as tf
import toolz
from evaluator20 import RecallEvaluator
from sampler import WarpSampler
import Dataset1 as Dataset
from tensorflow.contrib.layers.python.layers import regularizers
import scipy as sp
import os, sys
import argparse
from time import time

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    # name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            # with tf.variable_scope(name, *args, **kwargs):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class DMRL(object):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_dim=20,
                 batch_size = 10,
                 imagefeatures=None,
                 textualfeatures=None,
                 decay_r = 1e-4,
                 decay_c = 1e-3,
                 master_learning_rate=0.1,
                 hidden_layer_dim_a=256,
                 hidden_layer_dim_b=256,
                 dropout_rate_a=0.2,
                 dropout_rate_b=0.2,
                 ):
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        if imagefeatures is not None:
            self.imagefeatures = tf.constant(imagefeatures, dtype=tf.float32)
        else:
            self.imagefeatures = None
        if textualfeatures is not None:
            self.textualfeatures = tf.constant(textualfeatures, dtype=tf.float32)
        else:
            self.textualfeatures = None
        self.master_learning_rate = master_learning_rate
        self.hidden_layer_dim_a = hidden_layer_dim_a
        self.hidden_layer_dim_b = hidden_layer_dim_b
        self.dropout_rate_a = dropout_rate_a
        self.dropout_rate_b = dropout_rate_b
        self.n_factors = args.n_factors
        self.decay_r = decay_r
        self.decay_c = decay_c
        self.num_neg = args.num_neg
        self.user_positive_items_pairs = tf.placeholder(tf.int32, [self.batch_size, 2])
        self.negative_samples = tf.placeholder(tf.int32, [self.batch_size,self.num_neg])
        self.score_user_ids = tf.placeholder(tf.int32, [None])
        self.max_train_count = tf.placeholder(tf.int32, None)
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.user_embeddings
        self.item_embeddings
        self.feature_projection_visual
        self.feature_projection_textual
        self.embedding_loss
        self.loss
        self.optimize


    @define_scope
    def user_embeddings(self):
        return tf.Variable(self.initializer([self.n_users, self.embed_dim]),
                                                        name='user_embedding')
    @define_scope
    def item_embeddings(self):
        return tf.Variable(self.initializer([self.n_items, self.embed_dim]),
                           name='item_embedding')

    @define_scope
    def embedding_loss(self):
        """
        :return: the distance metric loss
        """
        # Let
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair
        # user embedding (N, K)
        users = tf.nn.embedding_lookup(self.user_embeddings,
                                       self.user_positive_items_pairs[:, 0],
                                       name="users")
        pos_items = tf.nn.embedding_lookup(self.item_embeddings, self.user_positive_items_pairs[:, 1],
                                           name="pos_items")
        pos_i_f = tf.nn.embedding_lookup(self.feature_projection_textual, self.user_positive_items_pairs[:, 1])
        pos_i_v = tf.nn.embedding_lookup(self.feature_projection_visual, self.user_positive_items_pairs[:, 1])

        # negative item embedding (N, K)
        neg_items = tf.reshape(tf.nn.embedding_lookup(self.item_embeddings, self.negative_samples, name="neg_items"), [-1,self.embed_dim])
        neg_i_f = tf.reshape(tf.nn.embedding_lookup(self.feature_projection_textual, self.negative_samples), [-1,self.embed_dim])
        neg_i_v = tf.reshape(tf.nn.embedding_lookup(self.feature_projection_visual, self.negative_samples), [-1,self.embed_dim])

        items = tf.concat([pos_items, neg_items], 0)
        textual_f = tf.concat([pos_i_f, neg_i_f], 0)
        visual_f = tf.concat([pos_i_v, neg_i_v], 0)

        user_a = tf.tile(users,[self.num_neg+1,1])
        user_factor_embedding = tf.split(users, self.n_factors, 1)
        item_factor_embedding = tf.split(items, self.n_factors, 1)
        item_factor_embedding_p = tf.split(pos_items, self.n_factors, 1)

        textual_factor_embedding = tf.split(textual_f, self.n_factors, 1)
        textual_factor_embedding_p = tf.split(pos_i_f, self.n_factors, 1)
        visual_factor_embedding = tf.split(visual_f, self.n_factors, 1)
        visual_factor_embedding_p = tf.split(pos_i_v, self.n_factors, 1)

        cor_loss = tf.constant(0, dtype=tf.float32)

        for i in range(0, self.n_factors - 1):
            x = visual_factor_embedding_p[i]
            y = visual_factor_embedding_p[i + 1]
            cor_loss += self._create_distance_correlation(x, y)
            x = textual_factor_embedding_p[i]
            y = textual_factor_embedding_p[i + 1]
            cor_loss += self._create_distance_correlation(x, y)
            x = user_factor_embedding[i]
            y = user_factor_embedding[i + 1]
            cor_loss += self._create_distance_correlation(x, y)
            x = item_factor_embedding_p[i]
            y = item_factor_embedding_p[i + 1]
            cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= ((self.n_factors + 1.0) * self.n_factors / 2)

        p_item, n_item = tf.split(items, [self.batch_size, self.num_neg*self.batch_size], 0)
        user_ap, user_an = tf.split(user_a, [self.batch_size, self.num_neg*self.batch_size], 0)

        user_factor_embedding_a = tf.split(user_a, self.n_factors, 1)
        user_factor_embedding_ap = tf.split(user_ap, self.n_factors, 1)
        user_factor_embedding_an = tf.split(user_an, self.n_factors, 1)

        p_item_factor_embedding = tf.split(p_item, self.n_factors, 1)
        n_item_factor_embedding = tf.split(n_item, self.n_factors, 1)

        regularizer = tf.constant(0, dtype=tf.float32)

        pos_scores, neg_scores = [], []


        for i in range(0, self.n_factors):
            weights = self._create_weight(user_factor_embedding_a[i], item_factor_embedding[i],
                                          textual_factor_embedding[i], visual_factor_embedding[i])
            p_weights, n_weights = tf.split(weights, [self.batch_size, self.num_neg*self.batch_size], 0)
            textual_trans = textual_factor_embedding[i]
            p_textual_trans, n_textual_trans = tf.split(textual_trans, [self.batch_size, self.num_neg*self.batch_size], 0)
            visual_trans = visual_factor_embedding[i]
            p_visual_trans, n_visual_trans = tf.split(visual_trans, [self.batch_size, self.num_neg*self.batch_size], 0)


            p_score = p_weights[:,1]*tf.nn.softplus(tf.reduce_sum(tf.multiply(user_factor_embedding_ap[i],
                                                p_textual_trans),1)) + p_weights[:,2]*tf.nn.softplus(tf.reduce_sum(tf.multiply(user_factor_embedding_ap[i],p_visual_trans),1)) +p_weights[:,0]*tf.nn.softplus(tf.reduce_sum(tf.multiply(
                                                    user_factor_embedding_ap[i], p_item_factor_embedding[i]),1))

            pos_scores.append(tf.expand_dims(p_score,1))

            n_score = n_weights[:,1]*tf.nn.softplus(tf.reduce_sum(tf.multiply(user_factor_embedding_an[i],
                                                n_textual_trans),1)) + n_weights[:,2]*tf.nn.softplus(tf.reduce_sum(tf.multiply(user_factor_embedding_an[i],n_visual_trans),1)) +n_weights[:,0]*tf.nn.softplus(tf.reduce_sum(tf.multiply(
                                                    user_factor_embedding_an[i], n_item_factor_embedding[i]),1))

            neg_scores.append(tf.expand_dims(n_score,1))

        pos_s = tf.concat(pos_scores, 1)
        neg_s = tf.concat(neg_scores, 1)

        regularizer += tf.nn.l2_loss(users) + tf.nn.l2_loss(
            pos_items) + tf.nn.l2_loss(neg_items) + tf.nn.l2_loss(pos_i_v) + tf.nn.l2_loss(neg_i_v) + tf.nn.l2_loss(pos_i_f) + tf.nn.l2_loss(neg_i_f)

        regularizer = regularizer / self.batch_size

        pos_score = tf.reduce_sum(pos_s, 1, name="pos")

        negtive_score = tf.reduce_max(tf.reshape(tf.reduce_sum(neg_s, 1),[self.batch_size,self.num_neg]),1)

        loss_per_pair = tf.nn.softplus(-(pos_score - negtive_score))

        loss = tf.reduce_sum(loss_per_pair, name="loss")

        return loss + self.decay_r*regularizer + self.decay_c*cor_loss


    def _create_weight(self,user,item,textual,visual):
        input = tf.nn.l2_normalize(tf.concat([user,item,textual,visual],1),1)
        output_h = tf.layers.dense(inputs=input, units=3, activation=tf.nn.tanh, name="weight_h",reuse=tf.AUTO_REUSE)
        output = tf.layers.dense(inputs=output_h, units=3, activation=None, use_bias=None, name="weight_o",reuse=tf.AUTO_REUSE)
        return tf.nn.softmax(output,1)

    @define_scope
    def feature_projection_visual(self):
        """
        :return: the projection of the feature vectors to the user-item embedding
        """
        mlp_layer_1 = tf.layers.dense(
            inputs=tf.nn.l2_normalize(self.imagefeatures,1),
            units=2*self.hidden_layer_dim_a,
            activation=tf.nn.leaky_relu,name="mlp_layer_v1")
        dropout = tf.layers.dropout(mlp_layer_1, self.dropout_rate_a)
        output = tf.layers.dense(inputs=tf.nn.l2_normalize(dropout,1), activation=None, units=self.embed_dim,reuse=tf.AUTO_REUSE,name="mlp_layer_v4")
        return output

    @define_scope
    def feature_projection_textual(self):
        """
        :return: the projection of the feature vectors to the user-item embedding
        """
        mlp_layer_1 = tf.layers.dense(
            inputs=tf.nn.l2_normalize(self.textualfeatures,1),
            units=2*self.hidden_layer_dim_b,
            activation=tf.nn.leaky_relu,name="mlp_layer_t1")
        dropout = tf.layers.dropout(mlp_layer_1, self.dropout_rate_a)
        output = tf.layers.dense(inputs=tf.nn.l2_normalize(dropout,1), activation=None, units=self.embed_dim,reuse=tf.AUTO_REUSE, name="mlp_layer_t4")
        return output

    def _create_distance_correlation(self, X1, X2):
        def _create_centered_distance(X):
            '''
                Used to calculate the distance matrix of N samples.
                (However how could tf store a HUGE matrix with the shape like 70000*70000*4 Bytes????)
            '''
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            # X = tf.math.l2_normalize(XX, axis=1)
            r = tf.reduce_sum(tf.square(X), 1, keepdims=True)
            D = tf.sqrt(tf.maximum(r - 2 * tf.matmul(a=X, b=X, transpose_b=True) + tf.transpose(r), 0.0) + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - tf.reduce_mean(D, axis=0, keepdims=True) - tf.reduce_mean(D, axis=1, keepdims=True) \
                + tf.reduce_mean(D)
            return D

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = tf.cast(tf.shape(D1)[0], tf.float32)
            dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2) / (n_samples * n_samples), 0.0) + 1e-8)
            # dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2)) / n_samples
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        dcor = dcov_12 / (tf.sqrt(tf.maximum(dcov_11 * dcov_22, 0.0)) + 1e-10)
        # return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor

    @define_scope
    def loss(self):
        loss = self.embedding_loss
        return loss

    @define_scope
    def clip_by_norm_op(self):
        return [tf.assign(self.user_embeddings, tf.clip_by_norm(self.user_embeddings, 1.0, axes=[1])),
                tf.assign(self.item_embeddings, tf.clip_by_norm(self.item_embeddings, 1.0, axes=[1]))]
    @define_scope
    def optimize(self):
        # have two separate learning rates. The first one for user/item embedding is un-normalized.
        # The second one for feature projector NN is normalized by the number of items.
        return tf.train.AdamOptimizer(self.master_learning_rate).minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings])

    @define_scope
    def item_scores(self):
        # (N_USER_IDS, 1, K)
        users = tf.expand_dims(tf.nn.embedding_lookup(self.user_embeddings, self.score_user_ids), 1)

        # (1, N_ITEM, K)
        item = tf.expand_dims(self.item_embeddings, 0)
        textual = tf.expand_dims(self.feature_projection_textual, 0)
        visual = tf.expand_dims(self.feature_projection_visual, 0)

        item_expand = tf.reshape(tf.tile(item, [tf.shape(users)[0], 1, 1]), [-1, self.embed_dim])
        textual_expand = tf.reshape(tf.tile(textual, [tf.shape(users)[0], 1, 1]), [-1, self.embed_dim])
        visual_expand = tf.reshape(tf.tile(visual, [tf.shape(users)[0], 1, 1]), [-1, self.embed_dim])

        users_expand = tf.reshape(tf.tile(users, [1, tf.shape(item)[1], 1]), [-1, self.embed_dim])

        user_expand_factor_embedding = tf.split(users_expand, self.n_factors, 1)
        item_expand_factor_embedding = tf.split(item_expand, self.n_factors, 1)

        textual_expand_factor_embedding = tf.split(textual_expand, self.n_factors, 1)
        visual_expand_factor_embedding = tf.split(visual_expand, self.n_factors, 1)

        factor_scores = []
        factor_sc = []
        factor_ws = []
        for i in range(0, self.n_factors):
            weights = self._create_weight(user_expand_factor_embedding[i],item_expand_factor_embedding[i],textual_expand_factor_embedding[i],visual_expand_factor_embedding[i])
            textual_trans = textual_expand_factor_embedding[i]
            visual_trans = visual_expand_factor_embedding[i]
            f_score = weights[:,1]*tf.nn.softplus(tf.reduce_sum(tf.multiply(user_expand_factor_embedding[i],
                                                textual_trans),1))+ weights[:,2]*tf.nn.softplus(tf.reduce_sum(tf.multiply(user_expand_factor_embedding[i],visual_trans),1)) + weights[:,0]*tf.nn.softplus(tf.reduce_sum(tf.multiply(
                                                    user_expand_factor_embedding[i], item_expand_factor_embedding[i]),1))
            factor_scores.append(tf.expand_dims(f_score, 1))
            factor_sc.append([weights[:,0]*tf.nn.softplus(tf.reduce_sum(tf.multiply(
                                                    user_expand_factor_embedding[i], item_expand_factor_embedding[i]),1)),weights[:,1]*tf.nn.softplus(tf.reduce_sum(tf.multiply(user_expand_factor_embedding[i],
                                                textual_trans),1)),weights[:,2]*tf.nn.softplus(tf.reduce_sum(tf.multiply(user_expand_factor_embedding[i],visual_trans),1))])
            factor_ws.append(weights)

        factor_s = tf.concat(factor_scores, 1)
        scores = tf.reshape(tf.reduce_sum(factor_s, axis=1),[tf.shape(users)[0],-1])
        return scores,factor_sc,factor_ws

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def optimize(model, sampler, train, train_num, test):
    """
    Optimize the model. TODO: implement early-stopping
    :param model: model to optimize
    :param sampler: mini-batch sampler
    :param train: train user-item matrix
    :param valid: validation user-item matrix
    :return: None
    """
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # all test users to calculate recall validation
    test_users = np.asarray(list(set(test.nonzero()[0])),dtype=np.int32)
    EVALUATION_EVERY_N_BATCHES = train_num // args.batch_size + 1
    cur_best_pre_0 = 0.
    pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], []
    stopping_step = 0

    for epoch in range(args.epochs):
        t1 = time()
        # TODO: early stopping based on validation recall
        # train model
        losses =0
        # run n mini-batches
        for _ in range(10*EVALUATION_EVERY_N_BATCHES):
            user_pos, neg = sampler.next_batch()
            _, loss = sess.run((model.optimize, model.loss),
                               {model.user_positive_items_pairs: user_pos,
                                model.negative_samples: neg})
            losses+=loss
        t2 = time()

        testresult = RecallEvaluator(model, train, test)
        test_recalls = []
        test_ndcg = []
        test_hr = []
        test_pr = []

        for user_chunk in toolz.partition_all(20, test_users):
            recalls, ndcgs, hit_ratios, precisions = testresult.eval(sess, user_chunk)
            test_recalls.extend(recalls)
            test_ndcg.extend(ndcgs)
            test_hr.extend(hit_ratios)
            test_pr.extend(precisions)

        recalls = sum(test_recalls)/float(len(test_recalls))
        precisions = sum(test_pr)/float(len(test_pr))
        hit_ratios = sum(test_hr)/float(len(test_hr))
        ndcgs = sum(test_ndcg)/float(len(test_ndcg))

        rec_loger.append(recalls)
        pre_loger.append(precisions)
        ndcg_loger.append(ndcgs)
        hit_loger.append(hit_ratios)

        t3 = time()
        print("epochs%d  [%.1fs + %.1fs]: train loss=%.5f, result=recall:%.5f, pres:%.5f, hr:%.5f, ndcg:%.5f"%(epoch, t2 - t1, t3 - t2, losses/(10*EVALUATION_EVERY_N_BATCHES), recalls,precisions,hit_ratios,ndcgs))

        cur_best_pre_0, stopping_step, should_stop = early_stopping(recalls, cur_best_pre_0,stopping_step, expected_order='acc', flag_step=5)

        if should_stop == True:
            sampler.close()
            break
        if epoch == args.epochs - 1:
            sampler.close()

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs)
    idx = list(recs).index(best_rec_0)
    final_perf = "Best Iter = recall:%.5f, pres:%.5f, hr:%.5f, ndcg:%.5f"%(recs[idx], pres[idx], hit[idx], ndcgs[idx])
    print(final_perf)

def parse_args():
    parser = argparse.ArgumentParser(description='Run DMRL.')
    parser.add_argument('--dataset', nargs='?',default='Office', help='Choose a dataset.')
    parser.add_argument('--epochs', type=int,default=1000, help = 'total_epochs')
    parser.add_argument('--gpu', nargs='?',default='1', help = 'gpu_id')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate.')
    parser.add_argument('--decay_r', type=float, default=1e-2, help='decay_r.')
    parser.add_argument('--decay_c', type=float, default=1e-0, help='decay_c.')
    parser.add_argument('--decay_p', type=float, default=0, help='decay_p.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--n_factors', type=int, default=4,help='Number of factors.')
    parser.add_argument('--num_neg', type=int,default=4, help = 'negative items')
    parser.add_argument('--hidden_layer_dim_a', type=int, default=256, help='Hidden layer dim a.')
    parser.add_argument('--hidden_layer_dim_b', type=int, default=128, help='Hidden layer dim b.')
    parser.add_argument('--dropout_a', type=float, default=0.2, help='dropout_a.')
    parser.add_argument('--dropout_b', type=float, default=0.2, help='dropout_b.')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # get user-item matrix
    # make feature as dense matrix
    args = parse_args()
    Filename = args.dataset
    Filepath = 'Data/' + Filename
    # get train/valid/test user-item matrices
    dataset = Dataset.Dataset(Filepath)
    train, test = dataset.trainMatrix, dataset.testRatings
    textualfeatures, imagefeatures = dataset.textualfeatures, dataset.imagefeatures
    n_users, n_items = max(train.shape[0],test.shape[0]),max(train.shape[1],test.shape[1])
    train_num = dataset.train_num
    # create warp sampler
    sampler = WarpSampler(train, batch_size=args.batch_size, n_negative=args.num_neg, check_negative=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = DMRL(n_users,
                n_items,
                # enable feature projection
                imagefeatures=imagefeatures,
                textualfeatures=textualfeatures,
                embed_dim=128,
                batch_size=args.batch_size,
                master_learning_rate=args.learning_rate,
                # the size of the hidden layer in the feature projector NN
                hidden_layer_dim_a=args.hidden_layer_dim_a,
                hidden_layer_dim_b=args.hidden_layer_dim_b,
                decay_r = args.decay_r,
                decay_c = args.decay_c,
                dropout_rate_a=args.dropout_a,
                dropout_rate_b=args.dropout_b,
                )

    optimize(model, sampler, train, train_num, test)
