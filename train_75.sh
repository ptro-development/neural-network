#!/bin/bash

ARGS=($( cat 50-trained-networks.txt | grep net | grep =\"/run | cut -d '"' -f 2 | sed s/\\/run\\/media\\/pwrap\\/187675E57675C452\\/50-trained/50-trained-networks/ ))

./training_sequence_75.sh 75 training_data_75.json ${ARGS[0]} ${ARGS[1]} ${ARGS[2]} ${ARGS[3]} ${ARGS[4]} ${ARGS[5]} ${ARGS[6]} ${ARGS[7]}
