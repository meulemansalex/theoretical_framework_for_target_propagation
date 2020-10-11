#!/usr/bin/env python3
# Copyright 2019 Alexander Meulemans
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import importlib
import argparse
import os
import sys
import numpy as np
import main


def _override_cmd_arg(config):
    sys.argv = [sys.argv[0]]
    for k, v in config.items():
        if isinstance(v, bool):
            cmd = '--%s' % k if v else ''
        else:
            cmd = '--%s=%s' % (k, str(v))
        if not cmd == '':
            sys.argv.append(cmd)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_module', type=str,
                        default='figure_scripts.config_toy_examples',
                        help='The name of the module containing the config.')
    args = parser.parse_args()
    config_module = importlib.import_module(args.config_module)
    _override_cmd_arg(config_module.config)
    summary = main.run()
    return summary

if __name__ == '__main__':
    run()