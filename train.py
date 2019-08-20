# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.7.14
# state: modified


import argparse
import yaml
from easydict import EasyDict
from trainer.trainer import Trainer

# parameters path
parser = argparse.ArgumentParser(description='Tensorflow Classification Proj Training')
parser.add_argument('--config-path', default='./exp_configs/captcha_simple_model/', type=str)
args = parser.parse_args()

# read config parameters from yaml file
config_path = args.config_path
output_path = config_path.replace("exp_configs", "exp_output")
with open(config_path + '/config.yaml') as f:
    config = yaml.load(f)
# convert to dict
config = EasyDict(config)

if __name__ == "__main__":
    trainer = Trainer(output_path=output_path, config=config)
    trainer.start_train_and_val()
