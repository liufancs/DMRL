import math
import heapq
import numpy as np
import tensorflow as tf
from scipy.sparse import lil_matrix
from sklearn.metrics import roc_auc_score

class RecallEvaluator(object):
    def __init__(self, model, train_user_item_matrix,test_user_item_matrix ):
        """
        Create a evaluator for recall@K evaluation
        :param model: the model we are going to evaluate
        :param train_user_item_matrix: the user-item pairs used in the training set. These pairs will be ignored
               in the recall calculation
        :param test_user_item_matrix: the held-out user-item pairs we make prediction against
        """
        self.model = model
        self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)

        self.n_users = max(train_user_item_matrix.shape[0],test_user_item_matrix.shape[0])
        self.n_items = max(train_user_item_matrix.shape[1],test_user_item_matrix.shape[1])

        self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u])
                                 for u in range(test_user_item_matrix.shape[0]) if self.test_user_item_matrix.rows[u]}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                      for u in range(train_user_item_matrix.shape[0]) if self.train_user_item_matrix.rows[u]}
            self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
        else:
            self.max_train_count = 0


    def eval(self, sess, users):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        :param users: the users to eval the recall
        :param k: compute the recall for the top K items
        :return: hitratio,ndgg@K
        """
        recalls = []
        precisions = []
        hit_ratios = []
        ndcgs = []
        user_tops,scores_s,scores_w = sess.run(self.model.item_scores,
                                {self.model.score_user_ids: users})

        user_batch_rating_uid = zip(users,user_tops)
        for user_id, tops in user_batch_rating_uid:
            batch_result = self.test_one_user(user_id,tops)
        recalls.append(batch_result['recall'])
        precisions.append(batch_result['precision'])
        hit_ratios.append(batch_result['hit_ratio'])
        ndcgs.append(batch_result['ndcg'])
        return recalls,ndcgs,hit_ratios,precisions

    def test_one_user(self, u,rating):

        # user u's ratings for user u
        rating = rating
        # uid
        u = u
        # user u's items in the training set
        training_items = self.user_to_train_set.get(u, set())
        # user u's items in the test set
        user_pos_test = self.user_to_test_set.get(u, set())

        all_items = set(range(self.n_items))

        test_items = list(all_items - set(training_items))

        r, auc = self.ranklist_by_sorted(user_pos_test, test_items, rating)

        return self.get_performance(user_pos_test, r, auc)

    def get_performance(self, user_pos_test, r, auc):
        K = 20
        precision=self.precision_at_k(r, K)
        recall=self.recall_at_k(r, K, len(user_pos_test))
        ndcg=self.ndcg_at_k(r, K)
        hit_ratio=self.hit_at_k(r, K)
        return {'recall': np.array(recall), 'precision': np.array(precision),
                'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

    def ranklist_by_sorted(self, user_pos_test, test_items, rating):
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max_item_score = heapq.nlargest(20, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = self.get_auc(item_score, user_pos_test)
        return r, auc

    def get_auc(self, item_score, user_pos_test):
        item_score = sorted(item_score.items(), key=lambda kv: kv[1])
        item_score.reverse()
        item_sort = [x[0] for x in item_score]
        posterior = [x[1] for x in item_score]

        r = []
        for i in item_sort:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = self.auc(ground_truth=r, prediction=posterior)
        return auc

    def auc(self, ground_truth, prediction):
        try:
            res = roc_auc_score(y_true=ground_truth, y_score=prediction)
        except Exception:
            res = 0.
        return res

    def precision_at_k(self, r, k):
        """Score is precision @ k
        Relevance is binary (nonzero is relevant).
        Returns:
            Precision @ k
        Raises:
            ValueError: len(r) must be >= k
        """
        assert k >= 1
        r = np.asarray(r)[:k]
        return np.mean(r)

    def dcg_at_k(self, r, k, method=1):
        """Score is discounted cumulative gain (dcg)
        Relevance is positive real values.  Can use binary
        as the previous methods.
        Returns:
            Discounted cumulative gain
        """
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')
        return 0.

    def ndcg_at_k(self, r, k, method=1):
        """Score is normalized discounted cumulative gain (ndcg)
        Relevance is positive real values.  Can use binary
        as the previous methods.
        Returns:
            Normalized discounted cumulative gain
        """
        dcg_max = self.dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return self.dcg_at_k(r, k, method) / dcg_max

    def recall_at_k(self, r, k, all_pos_num):
        r = np.asfarray(r)[:k]
        return np.sum(r) / all_pos_num

    def hit_at_k(self, r, k):
        r = np.array(r)[:k]
        if np.sum(r) > 0:
            return 1.
        else:
            return 0.