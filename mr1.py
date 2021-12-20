#!/usr/bin/env python
# coding=utf-8
from mrjob.job import MRJob
from mrjob.step import MRStep





class Step1(MRJob):
    """ 
    第一步是聚合单个用户的下的所有评分数据
    格式为：user_id, (item_count, rating_sum, [(item_id,rating)...])
    """
    def group_by_user_rating(self, key, line):
        """
        该mapper输出为：
        17 70,3
        35 21,1
        49 19,2
        49 21,1
        """
        user_id, item_id, rating = line.split('|')
        yield user_id, (item_id, float(rating))
 
    def count_ratings_users_freq(self, user_id, values):
        """
        该reducer输出为：
        49 (3,7,[19,2 21,1 70,4])
        """
        item_count = 0
        item_sum = 0
        final = []
        for item_id, rating in values:
            item_count += 1
            item_sum += rating
            final.append((item_id, rating))
 
        yield user_id, (item_count, item_sum, final)
 
    def steps(self):
        return [MRStep(mapper=self.group_by_user_rating,
                        reducer=self.count_ratings_users_freq),]
 
if __name__ == '__main__':
    
    Step1.run()