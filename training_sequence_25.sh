#!/bin/bash

#$1 - some name
#$2 - data file

. base.sh

SCRIPT_NAME="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
log_command "${SCRIPT_NAME} $@"
log_machine_id

for (( RUN=0; RUN < ${SETS}; ++RUN ))
do

    clean_command_file
    LAYERS="4,6,4"
    LOG="/tmp/results_$1.txt"
    init_network "${LAYERS}"

    clean_command_file
    LAYERS="4,6,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    ALL_EPOCHS=25000
    TEST_CASE="1. Starting from ${S_LAYERS} network without change, a lower bound ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    static_network_test "${TEST_CASE}" "${LAYERS}"

    clean_command_file
    LAYERS="4,6,4"
    NEW_LAYERS="4,6,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    EPOCH_LAYERS_ADDED=7500
    ALL_EPOCHS=25000
    TEST_CASE="2. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with old neurons locked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":2,"layers_neurons":[7, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[true,true,[true,true,true,true,false,false,false],false],"conditional":true,"condition_epoch>":EPOCH},{"command":"train_from_snapshot","position":1,"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    LAYERS="4,6,4"
    NEW_LAYERS="4,6,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    EPOCH_LAYERS_ADDED=15000
    ALL_EPOCHS=25000
    TEST_CASE="3. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with old neurons locked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":2,"layers_neurons":[7, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[true,true,[true,true,true,true,false,false,false],false],"conditional":true,"condition_epoch>":EPOCH},{"command":"train_from_snapshot","position":1,"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    LAYERS="4,6,4"
    NEW_LAYERS="4,6,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    EPOCH_LAYERS_ADDED=7500
    ALL_EPOCHS=25000
    LOCKED_LAYERS="4|6"
    TEST_CASE="4. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with layers ${LOCKED_LAYERS} locked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":2,"layers_neurons":[7, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[true,true,false,false],"conditional":true,"condition_epoch>":EPOCH},{"command":"train_from_snapshot","position":1,"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    LAYERS="4,6,4"
    NEW_LAYERS="4,6,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    EPOCH_LAYERS_ADDED=15000
    ALL_EPOCHS=25000
    LOCKED_LAYERS="4|6"
    TEST_CASE="5. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with layers ${LOCKED_LAYERS} locked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":2,"layers_neurons":[7, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[true,true,false,false],"conditional":true,"condition_epoch>":EPOCH},{"command":"train_from_snapshot","position":1,"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    LAYERS="4,6,4"
    NEW_LAYERS="4,6,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    EPOCH_LAYERS_ADDED=7500
    ALL_EPOCHS=25000
    TEST_CASE="6. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with all neurons unlocked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":2,"layers_neurons":[7, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[false,false,false,false],"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    LAYERS="4,6,4"
    NEW_LAYERS="4,6,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    EPOCH_LAYERS_ADDED=15000
    ALL_EPOCHS=25000
    TEST_CASE="7. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with all neurons unlocked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":2,"layers_neurons":[7, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[false,false,false,false],"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    LAYERS="4,6,4"
    NEW_LAYERS="4,13,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    ALL_EPOCHS=25000
    EPOCH_LAYERS_ADDED=7500
    TEST_CASE="8. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with all neurons unlocked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":1,"layers_neurons":[13, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[false,false,false],"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    LAYERS="4,6,4"
    NEW_LAYERS="4,13,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    ALL_EPOCHS=25000
    EPOCH_LAYERS_ADDED=15000
    TEST_CASE="9. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with all neurons unlocked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":1,"layers_neurons":[13, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[false,false,false],"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    NEW_LAYERS="4,6,7,4"
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    ALL_EPOCHS=25000
    init_network "${NEW_LAYERS}"
    TEST_CASE="10. Starting from ${S_NEW_LAYERS} network without change, an upper bound ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    static_network_test "${TEST_CASE}" "${NEW_LAYERS}"

    clean_command_file
    NEW_LAYERS="4,13,4"
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    ALL_EPOCHS=25000
    init_network "${NEW_LAYERS}"
    TEST_CASE="11. Starting from ${S_NEW_LAYERS} network without change, an upper bound ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    static_network_test "${TEST_CASE}" "${NEW_LAYERS}"

    clean_command_file
done
