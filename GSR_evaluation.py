import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, Conversation, SeparatorStyle

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Data Preprocessing
# ========================================


# ========================================
#             Model Evaluation
# ========================================
CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. "
           "Please answer my questions. ",
           # "Human: <Img><ImageHere></Img> What are the event verb and roles nouns in this image? ###"
           # "Assistant: 'verb': 'racing', 'nouns': {'place': 'racetrack', 'agent': 'racer'}. ###"
           # "Human: <Img><ImageHere></Img> What are the event verb and roles nouns in this image? ###"
           # "Assistant: 'verb': 'educating', 'nouns': {'place': 'classroom', 'teacher': 'woman', 'student': 'child', 'subject': 'mathematics'}. ###"
           # "Human: <Img><ImageHere></Img> What are the event verb and roles nouns in this image? ###"
           # "Assistant: 'verb': 'hugging', 'nouns': {'agent': 'man', 'hugged': 'woman'}. ###"
           # "Human: <Img><ImageHere></Img> What are the event verb and roles nouns in this image? ###"
           # "Assistant: 'verb': 'eating', 'nouns': {'food': 'dog food', 'place': 'floor', 'container': 'bowl', 'agent': 'dog', 'tool': 'mouth'}. ###"
           # "Human: <Img><ImageHere></Img> What are the event verb and roles nouns in this image? ###"
           # "Assistant: 'verb': 'drinking', 'nouns': {'place': 'table', 'container': 'glass', 'liquid': 'lemonade', 'agent': 'child'}. ",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

example_chat_state = CONV_VISION.copy()
example_images = ['./dataset/SWIG/racing_100.jpg',
                  './dataset/SWIG/educating_41.jpg',
                  './dataset/SWIG/hugging_85.jpg',
                  './dataset/SWIG/eating_300.jpg',
                  './dataset/SWIG/drinking_58.jpg',
                  ]
example_img_embed_list = []
# for example_image in example_images:
#     chat.upload_img(example_image, example_chat_state, example_img_embed_list)

num_beams = 1
temperature = 0.2


images = [
    './dataset/SWIG/educating_88.jpg',  # {"image_id": "educating_88.jpg", "caption": "{'frames': [{'place': 'school', 'teacher': 'woman', 'student': 'child', 'subject': 'reading'}, {'place': 'classroom', 'teacher': 'woman', 'student': 'girl', 'subject': 'finance'}, {'place': 'classroom', 'teacher': 'woman', 'student': 'girl', 'subject': ''}], 'verb': 'educating'}"}
    './dataset/SWIG/eating_401.jpg',  # {"image_id": "eating_401.jpg", "caption": "{'frames': [{'food': 'cake', 'place': 'inside', 'tool': 'fork', 'container': 'plate', 'agent': 'male child'}, {'food': 'cake', 'place': 'room', 'tool': 'fork', 'container': 'plate', 'agent': 'male child'}, {'food': 'cake', 'place': 'table', 'tool': 'fork', 'container': 'plate', 'agent': 'male child'}], 'verb': 'eating'}"}
    './dataset/SWIG/drinking_126.jpg',  # {"image_id": "drinking_126.jpg", "caption": "{'frames': [{'place': 'outdoors', 'container': 'water bottle', 'liquid': 'water', 'agent': 'girl'}, {'place': 'outdoors', 'container': 'bottle', 'liquid': 'water', 'agent': 'girl'}, {'place': 'outdoors', 'container': 'water bottle', 'liquid': 'water', 'agent': 'woman'}], 'verb': 'drinking'}"}
    './dataset/SWIG/calling_74.jpg',  # {"image_id": "calling_74.jpg", "caption": "{'frames': [{'tool': 'cellular telephone', 'place': 'outdoors', 'agent': 'man'}, {'tool': 'cellular telephone', 'place': 'outdoors', 'agent': 'man'}, {'tool': 'cellular telephone', 'place': 'street', 'agent': 'man'}], 'verb': 'calling'}"}
    './dataset/SWIG/bowing_80.jpg',  # {"image_id": "bowing_80.jpg", "caption": "{'frames': [{'place': 'office', 'agent': 'businesswoman'}, {'place': 'office', 'agent': 'couple'}, {'place': 'office', 'agent': 'woman'}], 'verb': 'bowing'}"}
    './dataset/SWIG/carrying_19.jpg',  # {"image_id": "carrying_19.jpg", "caption": "{'frames': [{'item': 'banner', 'place': 'pier', 'agent': 'people', 'agentpart': ''}, {'item': 'poster', 'place': 'street', 'agent': 'people', 'agentpart': 'thorax'}, {'item': 'signboard', 'place': 'road', 'agent': 'people', 'agentpart': 'hand'}], 'verb': 'carrying'}"}
    './dataset/SWIG/arresting_67.jpg',  # {"image_id": "arresting_67.jpg", "caption": "{'frames': [{'place': 'outside', 'suspect': 'man', 'agent': 'policeman'}, {'place': 'street', 'suspect': 'young buck', 'agent': 'policeman'}, {'place': 'outside', 'suspect': 'Spanish American', 'agent': 'police'}], 'verb': 'arresting'}"}
    './dataset/SWIG/arresting_0.jpg',
    './dataset/SWIG/bathing_5.jpg',  # {"image_id": "bathing_5.jpg", "caption": "{'frames': [{'coagent': 'dog', 'substance': 'soap', 'place': '', 'tool': 'hand', 'agent': 'person'}, {'coagent': 'dog', 'substance': 'soap', 'place': '', 'tool': 'hand', 'agent': 'person'}, {'coagent': 'dog', 'substance': 'shampoo', 'place': 'bathtub', 'tool': '', 'agent': 'person'}], 'verb': 'bathing'}"}
    './dataset/SWIG/bathing_110.jpg',
    './dataset/SWIG/barbecuing_36.jpg',  # {"image_id": "barbecuing_36.jpg", "caption": "{'frames': [{'food': 'short ribs', 'place': 'outdoors', 'agent': 'man'}, {'food': 'sparerib', 'place': 'forest', 'agent': 'man'}, {'food': 'barbecued spareribs', 'place': 'outdoors', 'agent': 'man'}], 'verb': 'barbecuing'}"}
    './dataset/SWIG/barbecuing_82.jpg',
]
for image in images:
    chat_state = CONV_VISION.copy()
    img_embed_list = [img_embed for img_embed in example_img_embed_list]
    chat.upload_img(image, chat_state, img_embed_list)
    user_message = 'What are the event verb and roles nouns in this image?'
    chat.ask(user_message, chat_state)
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_embed_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    print(f"{image.split('/')[-1]}: {llm_message}")
