from challenge_1.models.convnext import ConvNext
from challenge_1.models.copilot_net import CopilotModel
from challenge_1.models.efficientnetb7 import EfficientNet
from challenge_1.models.xception import Xception

NET_TO_MODEL = {
    "copilotnet": CopilotModel,
    "efficientnet": EfficientNet,
    "xception": Xception,
    "convnext": ConvNext,
}
