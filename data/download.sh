#!/bin/bash
# File              : download.sh
# Author            : Jiacheng Chen <cjc0722hz@gmail.com>
# Date              : 05.11.2019
# Last Modified Date: 05.11.2019
# Last Modified By  : Jiacheng Chen <cjc0722hz@gmail.com>
# Description: download and unzip floor-sp's 100 public indoor scanning released by Beike
mkdir public_100
cd public_100
mkdir zips
mkdir raws
cd ..
python download_from_email.py
cd public_100
sh unzip_all.sh
cd ..
