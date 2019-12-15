#!/bin/bash

BIN="./nn.py"
TRAINING_DATA="$2"
NAME="$1"
START_EPOCH=0
ALL_EPOCHS=25000
SETS=4

LOG="/media/win-data/training/results_$1.txt"
rm -f ${LOG}
COMMAND_FILE="/media/win-data/training/$1_network_command"

function clean_command_file {
    rm -f ${COMMAND_FILE}
}

function init_network {
    INITIAL_NETWORK="/media/win-data/training/saved_network_$1_${NAME}.net"
    AT=`date`
    CASE="Initial network setup"
    echo "START - ${AT} - ${CASE}" >> ${LOG}
    ${BIN} -q -d ${TRAINING_DATA} -n 1 -e 50 -r 50 -s ${INITIAL_NETWORK} -g ${COMMAND_FILE} -k "$1" 2>&1 >> ${LOG}
    TO=`date`
    echo "END - ${TO} - ${CASE}" >> ${LOG}
}

function static_network_test {
    for (( try=0; try < ${SETS}; ++try ))
    do
        CASE="$1, Run ${try}"
        AT=`date`
        echo "START - ${AT} - ${CASE}" >> ${LOG}
        SAVED_NETWORK="/media/win-data/training/saved_network_$2_${NAME}_twm_${try}.net"
        echo "${BIN} -q -d ${TRAINING_DATA} -n 1 -e ${ALL_EPOCHS} -r 50 -l ${INITIAL_NETWORK} -s ${SAVED_NETWORK} -g ${COMMAND_FILE} -k \"$2\"" >> ${LOG}
        ${BIN} -q -d ${TRAINING_DATA} -n 1 -e ${ALL_EPOCHS} -r 50 -l ${INITIAL_NETWORK} -s ${SAVED_NETWORK} -g ${COMMAND_FILE} -k "$2" 2>&1 >> ${LOG}
        TO=`date`
        echo "END - ${TO} - ${CASE}" >> ${LOG}
    done
}

function added_layer_test {
    EPOCH=$(($4+$START_EPOCH))
    for (( try=0; try < ${SETS}; ++try ))
    do
        CASE="$2, ${EPOCH}, Run ${try}"
        AT=`date`
        echo "START - ${AT} - ${CASE}" >> ${LOG}
        SAVED_NETWORK=`echo "/media/win-data/training/saved_network_$3_${NAME}_$2_${EPOCH}_${try}.net" | sed s/[[:space:]]/-/g`
        echo "$1" | sed s/EPOCH/${EPOCH}/g  > ${COMMAND_FILE}
        cat ${COMMAND_FILE} >> ${LOG}
        echo "${BIN} -q -d ${TRAINING_DATA} -n 1 -e ${ALL_EPOCHS} -r 50 -l ${INITIAL_NETWORK} -s ${SAVED_NETWORK} -g ${COMMAND_FILE} -k \"$3\"" >> ${LOG}
        ${BIN} -q -d ${TRAINING_DATA} -n 1 -e ${ALL_EPOCHS} -r 50 -l ${INITIAL_NETWORK} -s ${SAVED_NETWORK} -g ${COMMAND_FILE} -k "$3" >> ${LOG}
        TO=`date`
        echo "END - ${TO} - ${CASE}" >> ${LOG}
    done
}

function log_machine_id {
    uname -a >> ${LOG}
    id >> ${LOG}
    cat /proc/cpuinfo >> ${LOG}
}

function log_command {
    echo "$@" >> ${LOG}
}
