# the relation consensus module by Bolei
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pdb

# this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
# 我认为的x-frames relation module应该是从num_frames中选择多个x-frames组合，无论是early fusion 或者
# late fusion，当然我更想试试late fusion，而非只是从一个视频中选x-frames做一次关系推理
class RelationModule(torch.nn.Module):
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.classifier = self.fc_fusion()

    # 将n个帧的特征连接按一般连接，后接两个FC layer，前一个用作关系推理，输出的中间特征为对这个动作的逻辑判断，后一个当做分类器
    def fc_fusion(self):
        num_bottleneck = 512
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck,self.num_class),
                )
        return classifier

    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        input = self.classifier(input)
        return input

# 为了节省时间，并不需要重新跑一遍MultiScale TRN，部分设定与其相同，仍然令segments=8, k=3
# 但是只关注3-frame relation, 分别做early fusion和late fusion来验证实验1的猜想
# early & late fusion vision
class RelationModuleWithFusion(torch.nn.Module):
    # 初始化中的设定不用改，可以沿用multi-scale中的配置
    # relation module in multi-scale with a classifier at the end
    def __init__(self, img_feature_dim, num_frames, num_class, fusion_type="early"):
        super(RelationModuleWithFusion, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)]
        self.fusion_type = fusion_type

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        self.classifier_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]

            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),# this is the newly added thing
                        nn.Linear(num_bottleneck, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),
                        )
            classifier = nn.Linear(num_bottleneck, self.num_class)
            self.fc_fusion_scales += [fc_fusion]
            self.classifier_scales += [classifier]
        # maybe we put another fc layer after the summed up results???
        print('Multi-Scale Temporal Relation with classifier in use')
        print(['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        scaleID = 5     # 由于scale从大到小排列，故scaleID=5 表示 3-frame relation
        idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]),
                                                      self.subsample_scales[scaleID], replace=False)
        if self.fusion_type == "early":
            feat_fusion = []
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                feat_fusion.append(act_relation)

            # 将3-frames scale下的多种组合得到的关系特征按元素相加求均值，最后整体用一个分类器分类
            early_fusion = torch.mean(torch.stack(feat_fusion, dim=1), dim=1)
            act_relation = self.classifier_scales[scaleID](early_fusion)

            return act_relation
        elif self.fusion_type == "late":
            act_all = torch.zeros(input.size(0), self.num_class)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += self.classifier_scales[scaleID](act_relation)

            return act_all

        else:
            raise ValueError('Unknown fusion type: ' + self.fusion_type)


    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


# Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]
# 我确定，论文中的公式表示early fusion，即对特征先相加，然后对最终的整体特征做分类
# 但是，代码中的公式表示late fusion，即对每一种scale都有不同的分类器，且对每一种scale中每一个实例都直接计算出分类得分
# 如果采用early fusion是否还能有如此效果？
# 当然，对于early fusion的使用，也不能说不同scale的关系特征随便按元素相加
class RelationModuleMultiScale(torch.nn.Module):

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        # 比如num_frames = 7, scales = [7, 6, 5, 4, 3, 2]
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        # relations_scales中记录了所有可能的组合情况，但并不是每一个都会用到，因为subsample_scales记录了会选用的个数
        # 比如，num_frames = 4时
        # relations_scales: [[(0, 1, 2, 3)], [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]]
        # subsample_scales: [1, 3, 3]
        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        # 不同的尺度对应不同的融合函数和不同的分类器，存放在ModuleList类型中
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Linear(num_bottleneck, self.num_class),
                        )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale，为什么要对最大scale融合做特殊处理
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


# 对相同scale的不同组合计算得出的关系特征做early fusion，也就是average pooling
class RelationModuleMultiScaleWithClassifier(torch.nn.Module):
    # relation module in multi-scale with a classifier at the end
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScaleWithClassifier, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] #

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        self.classifier_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]

            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),# this is the newly added thing
                        nn.Linear(num_bottleneck, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),
                        )
            classifier = nn.Linear(num_bottleneck, self.num_class)
            self.fc_fusion_scales += [fc_fusion]
            self.classifier_scales += [classifier]
        # maybe we put another fc layer after the summed up results???
        print('Multi-Scale Temporal Relation with classifier in use')
        print(['%d-frame relation' % i for i in self.scales])

    # 对于不同scale的关系特征，我猜应该是不能随意按元素相加的，但是对于同一个scale的多个关系特征按道理来说应该是可以的
    # 还有一个很重要的问题就是，github上有人做实验时发现，segments = 8时，single relation and multiple relation效果相差无几
    # 那么多scale是否是没有必要的，又或者说从侧面证明了，我们应该对不同scale对应的特征做加权后添加到3D CNN特征当中
    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)
        act_all = self.classifier_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]),
                                                          self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = self.classifier_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

        # for scaleID in range(1, len(self.scales)):
        #     # iterate over the scales
        #     idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
        #     early_fusion = []
        #     for idx in idx_relations_randomsample:
        #         act_relation = input[:, self.relations_scales[scaleID][idx], :]
        #         act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
        #         act_relation = self.fc_fusion_scales[scaleID](act_relation)
        #         early_fusion.append(act_relation)
        #
        #     # 将n-frames scale下的多种组合得到的关系特征按元素相加求均值，最后整体用一个分类器分类
        #     early_fusion = torch.mean(torch.stack(early_fusion, dim=1), dim=1)
        #     act_relation = self.classifier_scales[scaleID](early_fusion)
        #     act_all += act_relation
        # return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

def return_TRN(relation_type, img_feature_dim, num_frames, num_class):
    if relation_type == 'TRN':
        TRNmodel = RelationModule(img_feature_dim, num_frames, num_class)
    elif relation_type == 'TRNmultiscale':
        TRNmodel = RelationModuleMultiScale(img_feature_dim, num_frames, num_class)
    elif relation_type == "TRNmultiscaleWithClassifier":
        TRNmodel = RelationModuleMultiScaleWithClassifier(img_feature_dim, num_frames, num_class)
    elif relation_type == "TRNWithFusion":
        TRNmodel = RelationModuleWithFusion(img_feature_dim, num_frames, num_class, fusion_type="late")
    else:
        raise ValueError('Unknown TRN' + relation_type)


    return TRNmodel

if __name__ == "__main__":
    batch_size = 10
    num_frames = 8
    num_class = 174
    img_feature_dim = 256
    input_var = torch.randn(batch_size, num_frames, img_feature_dim)
    model = RelationModuleMultiScaleWithClassifier(img_feature_dim, num_frames, num_class)
    output = model(input_var)
    print(output.shape)


