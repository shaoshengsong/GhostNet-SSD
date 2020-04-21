# GhostNet-SSD
GhostNet Object Detection

GhostNet改造的目标检测


首先 git clone https://github.com/shaoshengsong/MobileNetV3-SSD-Compact-Version.git

任意选择一个文件
例如 model_v1(按照SSD原版设计).py 更名为model.py放到MobileNetV3-SSD-Compact-Version文件夹中

不同的是 先看看您使用的是哪个GPU,GhostNet用的是1

您还可以把 GhosetNet的AuxiliaryConvolutions 换成/MobileNetV3-SSD-Compact-Version里面的AuxiliaryConvolutions
GhosetNet的AuxiliaryConvolutions 是非常古老的写法.
代码里有不同的feature map您任选

其他操作步骤与MobileNetV3-SSD-Compact-Version一模一样.
