#!/bin/bash

ARGS=($( cat 75-trained-networks.txt | grep net | grep =\"/run | cut -d '"' -f 2 | sed s/\\/run\\/media\\/pwrap\\/187675E57675C452\\/75-trained/75-trained-networks/ ))

./training_sequence_100_2.sh 100 training_data_100.json ${ARGS[0]} ${ARGS[1]} ${ARGS[2]} ${ARGS[3]} ${ARGS[4]} ${ARGS[5]} ${ARGS[6]} ${ARGS[7]}
