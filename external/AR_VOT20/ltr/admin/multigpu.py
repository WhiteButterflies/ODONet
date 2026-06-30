import torch.nn as nn

'''原来的版本是根据“对象是否是nn.DataParallel类的实例”来判断是否使用了多GPU'''
# def is_multi_gpu(net):
#     return isinstance(net, (MultiGPU, nn.DataParallel))
#
#
# class MultiGPU(nn.DataParallel):
#     def __getattr__(self, item):
#         try:
#             return super().__getattr__(item)
#         except:
#             pass
#         return getattr(self.module, item)
'''由于我们使用的是nn.parallel.distributed.DistributedDataParallel
所以判断方式都要修改'''
def is_multi_gpu(net):
    return isinstance(net, (MultiGPU, nn.parallel.distributed.DistributedDataParallel))


class MultiGPU(nn.parallel.distributed.DistributedDataParallel):
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)