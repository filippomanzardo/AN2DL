from challenge_1.models.copilot_net import CopilotModel
from challenge_1.models.efficientnetb7 import EfficientNetB7

NET_TO_MODEL = {
    "copilotnet": CopilotModel,
    "efficientnetb7": EfficientNetB7,
}
