import torch
from torchinfo import summary
from torchvision import models

model_conv = models.resnet18(pretrained=True)
newmodel = torch.nn.Sequential(*(list(model_conv.children())[:-1]))
summary(newmodel)
summary(model_conv)











# state_dict = torch.load(
#     '/home/carlos/Downloads/gibson-2plus-mp3d-train-val-test-se-resneXt50-rgb.pth')
# model = torch.fr
# summary(model)
#
# self.device = (
#     torch.device("cuda:{}".format(config.PTH_GPU_ID))
#     if torch.cuda.is_available()
#     else torch.device("cpu")
# )
#
# actor_critic = PointNavResNetPolicy(
#     observation_space=observation_spaces,
#     action_space=action_spaces,
#     hidden_size=self.hidden_size,
#     normalize_visual_inputs="rgb" in spaces,
# )
# self.actor_critic.to(self.device)
#
# if config.MODEL_PATH:
#     ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
#     #  Filter only actor_critic weights
#     self.actor_critic.load_state_dict(
#         {  # type: ignore
#             k[len("actor_critic."):]: v
#             for k, v in ckpt["state_dict"].items()
#             if "actor_critic" in k
#         }
#     )
#
# else:
#     habitat.logger.error(
#         "Model checkpoint wasn't loaded, evaluating " "a random model."
#     )
