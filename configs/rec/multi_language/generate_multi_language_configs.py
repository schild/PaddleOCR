# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os.path
import logging
logging.basicConfig(level=logging.INFO)

support_list = {
    'it': 'italian',
    'xi': 'spanish',
    'pu': 'portuguese',
    'ru': 'russian',
    'ar': 'arabic',
    'ta': 'tamil',
    'ug': 'uyghur',
    'fa': 'persian',
    'ur': 'urdu',
    'rs': 'serbian latin',
    'oc': 'occitan',
    'rsc': 'serbian cyrillic',
    'bg': 'bulgarian',
    'uk': 'ukranian',
    'be': 'belarusian',
    'te': 'telugu',
    'ka': 'kannada',
    'chinese_cht': 'chinese tradition',
    'hi': 'hindi',
    'mr': 'marathi',
    'ne': 'nepali',
}

latin_lang = [
    'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
    'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
    'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
    'sw', 'tl', 'tr', 'uz', 'vi', 'latin'
]
arabic_lang = ['ar', 'fa', 'ug', 'ur']
cyrillic_lang = [
    'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
    'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic'
]
devanagari_lang = [
    'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
    'sa', 'bgc', 'devanagari'
]
multi_lang = latin_lang + arabic_lang + cyrillic_lang + devanagari_lang

assert (os.path.isfile("./rec_multi_language_lite_train.yml")
        ), "Loss basic configuration file rec_multi_language_lite_train.yml.\
You can download it from \
https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph/configs/rec/multi_language/"

global_config = yaml.load(
    open("./rec_multi_language_lite_train.yml", 'rb'), Loader=yaml.Loader)
project_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")
        self.add_argument(
            "-l",
            "--language",
            nargs='+',
            help=f"set language type, support {support_list}",
        )
        self.add_argument(
            "--train",
            type=str,
            help="you can use this command to change the train dataset default path"
        )
        self.add_argument(
            "--val",
            type=str,
            help="you can use this command to change the eval dataset default path"
        )
        self.add_argument(
            "--dict",
            type=str,
            help="you can use this command to change the dictionary default path"
        )
        self.add_argument(
            "--data_dir",
            type=str,
            help="you can use this command to change the dataset default root path"
        )

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        args.opt = self._parse_opt(args.opt)
        args.language = self._set_language(args.language)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

    def _set_language(self, type):
        lang = type[0]
        assert (type), "please use -l or --language to choose language type"
        assert (
            lang in support_list.keys() or lang in multi_lang
        ), f"the sub_keys(-l or --language) can only be one of support list: \n{multi_lang},\nbut get: {type}, please check your running command"
        if lang in latin_lang:
            lang = "latin"
        elif lang in arabic_lang:
            lang = "arabic"
        elif lang in cyrillic_lang:
            lang = "cyrillic"
        elif lang in devanagari_lang:
            lang = "devanagari"
        global_config['Global'][
            'character_dict_path'
        ] = f'ppocr/utils/dict/{lang}_dict.txt'
        global_config['Global']['save_model_dir'] = f'./output/rec_{lang}_lite'
        global_config['Train']['dataset']['label_file_list'] = [
            f"train_data/{lang}_train.txt"
        ]
        global_config['Eval']['dataset']['label_file_list'] = [
            f"train_data/{lang}_val.txt"
        ]
        global_config['Global']['character_type'] = lang
        assert (
            os.path.isfile(
                os.path.join(project_path, global_config['Global'][
                    'character_dict_path']))
        ), "Loss default dictionary file {}_dict.txt.You can download it from \
https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph/ppocr/utils/dict/".format(
            lang)
        return lang


def merge_config(config):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in config.items():
        if "." not in key:
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in global_config
            ), f"the sub_keys can only be one of global_config: {global_config.keys()}, but get: {sub_keys[0]}, please check your running command"
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]


def loss_file(path):
    assert os.path.exists(
        path
    ), f"There is no such file:{path},Please do not forget to put in the specified file"


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    merge_config(FLAGS.opt)
    save_file_path = f'rec_{FLAGS.language}_lite_train.yml'
    if os.path.isfile(save_file_path):
        os.remove(save_file_path)

    if FLAGS.train:
        global_config['Train']['dataset']['label_file_list'] = [FLAGS.train]
        train_label_path = os.path.join(project_path, FLAGS.train)
        loss_file(train_label_path)
    if FLAGS.val:
        global_config['Eval']['dataset']['label_file_list'] = [FLAGS.val]
        eval_label_path = os.path.join(project_path, FLAGS.val)
        loss_file(eval_label_path)
    if FLAGS.dict:
        global_config['Global']['character_dict_path'] = FLAGS.dict
        dict_path = os.path.join(project_path, FLAGS.dict)
        loss_file(dict_path)
    if FLAGS.data_dir:
        global_config['Eval']['dataset']['data_dir'] = FLAGS.data_dir
        global_config['Train']['dataset']['data_dir'] = FLAGS.data_dir
        data_dir = os.path.join(project_path, FLAGS.data_dir)
        loss_file(data_dir)

    with open(save_file_path, 'w') as f:
        yaml.dump(
            dict(global_config), f, default_flow_style=False, sort_keys=False)
    logging.info(f"Project path is          :{project_path}")
    logging.info("Train list path set to   :{}".format(global_config['Train'][
        'dataset']['label_file_list'][0]))
    logging.info("Eval list path set to    :{}".format(global_config['Eval'][
        'dataset']['label_file_list'][0]))
    logging.info("Dataset root path set to :{}".format(global_config['Eval'][
        'dataset']['data_dir']))
    logging.info("Dict path set to         :{}".format(global_config['Global'][
        'character_dict_path']))
    logging.info(
        f"Config file set to       :configs/rec/multi_language/{save_file_path}"
    )
