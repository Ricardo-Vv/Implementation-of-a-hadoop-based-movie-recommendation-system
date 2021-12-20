#!/usr/bin/env python
# -*- coding:utf-8 -*-
 
from mrjob.job import MRJob
 
class dataClean(MRJob):
    def mapper(self,key,line): #接收每一行的输入数据，处理后返回一堆key:value，初始化value值为1
        for line in sys.stdin:
            user_id, item_id, rating = line.split('|')  
            yield user_id, (item_id, float(rating)))
 
    def reducer(self, user_id, values): #接收mapper输出的key:value对进行整合，把相同key的value做累加（sum）操作后输出
        item_count = 0  
	    item_sum = 0  
        final = []
        for id, rating in values:  
            counts += 1  
            sums += rating  
            result.append((id, rating))  
    yield user_id, (item_count,sums,result)
 
if __name__ == '__main__':
    dataClean.run()