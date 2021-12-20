from mrjob.job import MRJob
from itertools import combinations
from math import sqrt
from mrjob.job import MRJob
from mrjob.step import MRStep

class Step2(MRJob):
    def pairwise_items(self,user_id,values):
        '''
        本mapper使用step1的输出作为输入，把user_id丢弃掉不再使用
        输出结果为 （item_1,item2），(rating_1,rating_2)
 
        这里combinations(iterable,number)的作用是求某个集合的组合，
        如combinations([1,2,3,4],2)就是在集合种找出任两个数的组合。
 
        这个mapper是整个任务的性能瓶颈，这是因为combinations函数生成的数据
        比较多，这么多的零散数据依次写回磁盘，IO操作过于频繁，可以用写一个
        Combiner来紧接着mapper做一些聚合操作（和Reducer相同），由Combiner
        把数据写回磁盘，该Combiner也可以用C库来实现，由Python调用。
        '''
        # 这里由于step1是分开的，把数据dump到文件result1.csv中，所以读取的时候
        # 需要按照字符串处理，如果step1和step2在同一个job内完成，则直接可以去掉
        # 这一行代码，在同一个job内完成参见steps函数的使用说明。
        # values = eval(values.split('\t')[1])
        item_count ,item_sum, ratings = values
        for item1,item2 in combinations(ratings,2):
            yield(item1[0],item2[0]),(item1[1],item2[1])

    def calculate_similarity(self,pair_key,lines):
        '''
        (Movie A,Movie B)作为Key，(A rating,B rating)作为该reducer的输入，
        每一次输入属于同一个用户，所有当两个key相同时，代表他们两个都看了A和B，所以
        按照这些所有都看了A、B的人的评分作为向量，计算A、B的皮尔逊系数。
        '''
        sum_xx,sum_xy,sum_yy,sum_x,sum_y,n=(0.0,0.0,0.0,0.0,0.0,0)
        item_pair,co_ratings = pair_key,lines
        item_xname,item_yname= item_pair
        for item_x,item_y in co_ratings:
            sum_xx+=item_x*item_x
            sum_yy+=item_y*item_y
            sum_xy+=item_x*item_y
            sum_y+=item_y
            sum_x+=item_x
            n+=1
        similarity=self.normalized_correlation(n,sum_xy,sum_x,sum_y,sum_xx,sum_yy)
        yield(item_xname,item_yname),(similarity,n)

    def steps(self):
        return[MRStep(mapper=self.pairwise_items,
                        reducer=self.calculate_similarity),]

    def normalized_correlation(self,n,sum_xy,sum_x,sum_y,sum_xx,sum_yy):
        numerator = (n*sum_xy-sum_x*sum_y)
        denominator = sqrt(n*sum_xx-sum_x*sum_x)*sqrt(n*sum_yy-sum_y*sum_y)
        similarity = numerator/denominator
        return similarity

if __name__ == "__main__":
    Step2.run()