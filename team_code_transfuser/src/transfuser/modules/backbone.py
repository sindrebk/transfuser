import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageCNN
from .lidar_encoder import LidarEncoder
from .gpt import GPT


def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x


class TransfuserBackbone(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.config = config

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))
        
        self.image_encoder = ImageCNN(architecture=image_architecture, normalize=True)

        if(config.use_point_pillars == True):
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len

        if(self.config.use_target_point_image == True):
            in_channels += 1

        self.lidar_encoder = LidarEncoder(architecture=lidar_architecture, in_channels=in_channels)

        self.transformer1 = GPT(n_embd=self.image_encoder.features.feature_info[1]['num_chs'],
                            n_head=config.n_head,
                            block_exp=config.block_exp,
                            n_layer=config.n_layer,
                            img_vert_anchors=config.img_vert_anchors,
                            img_horz_anchors=config.img_horz_anchors,
                            lidar_vert_anchors=config.lidar_vert_anchors,
                            lidar_horz_anchors=config.lidar_horz_anchors,
                            seq_len=config.seq_len,
                            embd_pdrop=config.embd_pdrop,
                            attn_pdrop=config.attn_pdrop,
                            resid_pdrop=config.resid_pdrop,
                            config=config, use_velocity=use_velocity)

        self.transformer2 = GPT(n_embd=self.image_encoder.features.feature_info[2]['num_chs'],
                            n_head=config.n_head,
                            block_exp=config.block_exp,
                            n_layer=config.n_layer,
                            img_vert_anchors=config.img_vert_anchors,
                            img_horz_anchors=config.img_horz_anchors,
                            lidar_vert_anchors=config.lidar_vert_anchors,
                            lidar_horz_anchors=config.lidar_horz_anchors,
                            seq_len=config.seq_len,
                            embd_pdrop=config.embd_pdrop,
                            attn_pdrop=config.attn_pdrop,
                            resid_pdrop=config.resid_pdrop,
                            config=config, use_velocity=use_velocity)

        self.transformer3 = GPT(n_embd=self.image_encoder.features.feature_info[3]['num_chs'],
                            n_head=config.n_head,
                            block_exp=config.block_exp,
                            n_layer=config.n_layer,
                            img_vert_anchors=config.img_vert_anchors,
                            img_horz_anchors=config.img_horz_anchors,
                            lidar_vert_anchors=config.lidar_vert_anchors,
                            lidar_horz_anchors=config.lidar_horz_anchors,
                            seq_len=config.seq_len,
                            embd_pdrop=config.embd_pdrop,
                            attn_pdrop=config.attn_pdrop,
                            resid_pdrop=config.resid_pdrop,
                            config=config, use_velocity=use_velocity)

        self.transformer4 = GPT(n_embd=self.image_encoder.features.feature_info[4]['num_chs'],
                            n_head=config.n_head,
                            block_exp=config.block_exp,
                            n_layer=config.n_layer,
                            img_vert_anchors=config.img_vert_anchors,
                            img_horz_anchors=config.img_horz_anchors,
                            lidar_vert_anchors=config.lidar_vert_anchors,
                            lidar_horz_anchors=config.lidar_horz_anchors,
                            seq_len=config.seq_len,
                            embd_pdrop=config.embd_pdrop,
                            attn_pdrop=config.attn_pdrop,
                            resid_pdrop=config.resid_pdrop,
                            config=config, use_velocity=use_velocity)

        if(self.image_encoder.features.feature_info[4]['num_chs'] != self.config.perception_output_features):
            self.change_channel_conv_image = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], self.config.perception_output_features, (1, 1))
            self.change_channel_conv_lidar = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], self.config.perception_output_features, (1, 1))
        else:
            self.change_channel_conv_image = nn.Sequential()
            self.change_channel_conv_lidar = nn.Sequential()

        # FPN fusion
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        # top down
        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        
        # lateral
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1, 1))
        
    def top_down(self, x):

        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        
        return p2, p3, p4, p5

    def forward(self, image, lidar, velocity):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''

        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image

        lidar_tensor = lidar

        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.act1(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)

        # Image fusion at (B, 72, 40, 176)
        # Lidar fusion at (B, 72, 64, 64)
        image_embd_layer1 = self.avgpool_img(image_features)
        lidar_embd_layer1 = self.avgpool_lidar(lidar_features)

        image_features_layer1, lidar_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1, velocity)
        image_features_layer1 = F.interpolate(image_features_layer1, size=(image_features.shape[2],image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, size=(lidar_features.shape[2],lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1

        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        # Image fusion at (B, 216, 20, 88)
        # Image fusion at (B, 216, 32, 32)
        image_embd_layer2 = self.avgpool_img(image_features)
        lidar_embd_layer2 = self.avgpool_lidar(lidar_features)
        image_features_layer2, lidar_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2, velocity)
        image_features_layer2 = F.interpolate(image_features_layer2, size=(image_features.shape[2],image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer2 = F.interpolate(lidar_features_layer2, size=(lidar_features.shape[2],lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2

        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        # Image fusion at (B, 576, 10, 44)
        # Image fusion at (B, 576, 16, 16)
        image_embd_layer3 = self.avgpool_img(image_features)
        lidar_embd_layer3 = self.avgpool_lidar(lidar_features)
        image_features_layer3, lidar_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3, velocity)
        image_features_layer3 = F.interpolate(image_features_layer3, size=(image_features.shape[2],image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer3 = F.interpolate(lidar_features_layer3, size=(lidar_features.shape[2],lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3

        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        # Image fusion at (B, 1512, 5, 22)
        # Image fusion at (B, 1512, 8, 8)
        image_embd_layer4 = self.avgpool_img(image_features)
        lidar_embd_layer4 = self.avgpool_lidar(lidar_features)

        image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4, velocity)
        image_features_layer4 = F.interpolate(image_features_layer4, size=(image_features.shape[2],image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer4 = F.interpolate(lidar_features_layer4, size=(lidar_features.shape[2],lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4

        # Downsamples channels to 512
        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)

        x4 = lidar_features
        image_features_grid = image_features  # For auxilliary information

        image_features = self.image_encoder.features.global_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)

        fused_features = image_features + lidar_features

        features = self.top_down(x4)
        return features, image_features_grid, fused_features
