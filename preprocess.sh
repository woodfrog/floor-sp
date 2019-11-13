#!/bin/bash
# File              : preprocess.sh
# Author            : Jiacheng Chen <cjc0722hz@gmail.com>
# Date              : 05.11.2019
# Last Modified Date: 05.11.2019
# Last Modified By  : Jiacheng Chen <cjc0722hz@gmail.com>

# Data pre-processing
cd utils
python data_preprocess.py
python data_writer.py
cd ..
