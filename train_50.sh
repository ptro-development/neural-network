#!/bin/bash

ARGS=($(cat 25-trained-networks.txt | grep net | grep =\"/tmp | cut -d '"' -f 2 | sed s/\\/tmp/25-trained-networks/ ))

./training_sequence_50.sh 50 training_data_50.json ${ARGS[0]} ${ARGS[1]} ${ARGS[2]} ${ARGS[3]} ${ARGS[4]} ${ARGS[5]} ${ARGS[6]} ${ARGS[7]}
