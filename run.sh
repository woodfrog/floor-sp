#!/bin/bash
# File              : run.sh
# Author            : Jiacheng Chen <cjc0722hz@gmail.com>
# Date              : 05.11.2019
# Last Modified Date: 05.11.2019
# Last Modified By  : Jiacheng Chen <cjc0722hz@gmail.com>

# Run mask-rcnn to get room segmentations, and prepare data for training corner/edgeness modules
cd mask-rcnn
python inference_corner.py
python inference_room.py
cd ..

# Run floop-sp, first run pre-trained corner/edgeness modules to get energy terms, then solve for the floorplan structure
# Remember to set up the weight path properly in floor-sp/configs/ before running every command
cd floor-sp
python mains/corner_main.py
python mains/associate_main.py
python mains/extract_floorplan.py
cd ..
