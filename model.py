import torch
import torch.nn.functional as F

class PoseNet(torch.nn.Module):

    def __init__(self, feature_extractor, num_features = 128, dropout = 0.5,track_running_status = False, pretrained = False):

        super(PoseNet, self).__init__()
        self.dropout = dropout
        self.track_running_status = track_running_status
        self.pretrained = pretrained
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc_in_features = self.feature_extractor.fc.in_features # in_features : size of input sample

        # translation

        self.fc_xy = torch.nn.Linear(num_features,2) # 2 outputs

        # initialization
        if self.pretrained:
            init_modules = [self.feature_extractor, self.fc_xy]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)

    def extract_features(self, x):
        x_features = self.feature_extractor(x)
        x_features = F.relu(x_features)
        if self.dropout > 0:
            x_features = F.dropout(x_features, p = self.dropout, training=self.training)
        return x_features

    def forward(self, x):

        # x is a batch of images : [batch_size * image, batch_size * image]

        if type(x) is list :

            x_features = [self.extract_features(xi) for xi in x]
            x_translations = [self.fc_xy(xi) for xi in x]
            return x_translations

        elif torch.is_tensor(x):

            x_features = self.extract_features(x)
            x_translations = self.fc_xy(x_features)
            return x_translations


class PoseNetCriterion(torch.nn.Module):

    def __init__(self, stereo = True, beta = 512.0, learn_beta = False, sx = 0.0, sq = -3.0):
        super(PoseNetCriterion, self).__init__()
        self.stereo = stereo
        self.loss_gn = torch.nn.MSELoss()
        self.learn_beta = learn_beta
        if not learn_beta:
            self.beta = beta
        else :
            self.beta = 1.0
        self.sx = torch.nn.Parameter(torch.Tensor([sx]), requires_grad = learn_beta)



    def forward(self, x, y):

        loss = 0
        if self.stereo :

            for i in range(2):
                # translation loss
                loss = loss + torch.exp(-self.sx) * self.loss_fn(x,y)

            loss = loss/2 # divided by 2 for comparing stereo vs non stereo mode

        else :
            loss = loss + torch.exp(-self.fx) * self.loss_fn(x,y)

        return loss







        