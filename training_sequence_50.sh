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
    LAYERS="4,6,7,4"
    LOG="/tmp/results_${1}.txt"
    init_network "${LAYERS}"

    clean_command_file
    LAYERS="4,6,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    ALL_EPOCHS=50000
    TEST_CASE="1. Starting from ${S_LAYERS} network without change, a lower bound ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    static_network_test "${TEST_CASE}" "${LAYERS}"

    clean_command_file
    INITIAL_NETWORK="${3}"
    LAYERS="4,6,7,4"
    NEW_LAYERS="4,6,7,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    EPOCH_LAYERS_ADDED=32500
    ALL_EPOCHS=25000
    TEST_CASE="2. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with old neurons locked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":3,"layers_neurons":[7, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[true,true,true,[true,true,true,true,false,false,false],false],"conditional":true,"condition_epoch>":EPOCH},{"command":"train_from_snapshot","position":2,"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    INITIAL_NETWORK="${4}"
    LAYERS="4,6,7,4"
    NEW_LAYERS="4,6,7,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    EPOCH_LAYERS_ADDED=40000
    ALL_EPOCHS=25000
    TEST_CASE="3. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with old neurons locked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":3,"layers_neurons":[7, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[true,true,true,[true,true,true,true,false,false,false],false],"conditional":true,"condition_epoch>":EPOCH},{"command":"train_from_snapshot","position":2,"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    INITIAL_NETWORK="${5}"
    LAYERS="4,6,7,4"
    NEW_LAYERS="4,6,7,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    EPOCH_LAYERS_ADDED=32500
    ALL_EPOCHS=25000
    LOCKED_LAYERS="4|6"
    TEST_CASE="4. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with layers ${LOCKED_LAYERS} locked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":3,"layers_neurons":[7, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[true,true,true,false,false],"conditional":true,"condition_epoch>":EPOCH},{"command":"train_from_snapshot","position":2,"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    INITIAL_NETWORK="${6}"
    LAYERS="4,6,7,4"
    NEW_LAYERS="4,6,7,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    EPOCH_LAYERS_ADDED=40000
    ALL_EPOCHS=25000
    LOCKED_LAYERS="4|6"
    TEST_CASE="5. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with layers ${LOCKED_LAYERS} locked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":3,"layers_neurons":[7, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[true,true,true,false,false],"conditional":true,"condition_epoch>":EPOCH},{"command":"train_from_snapshot","position":2,"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    INITIAL_NETWORK="${7}"
    LAYERS="4,6,7,4"
    NEW_LAYERS="4,6,7,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    EPOCH_LAYERS_ADDED=32500
    ALL_EPOCHS=25000
    TEST_CASE="6. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with all neurons unlocked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":3,"layers_neurons":[7, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[false,false,false,false,false],"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    INITIAL_NETWORK="${8}"
    LAYERS="4,6,7,4"
    NEW_LAYERS="4,6,7,7,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    EPOCH_LAYERS_ADDED=40000
    ALL_EPOCHS=25000
    TEST_CASE="7. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with all neurons unlocked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":3,"layers_neurons":[7, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[false,false,false,false,false],"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    INITIAL_NETWORK="${9}"
    LAYERS="4,13,4"
    NEW_LAYERS="4,14,6,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    ALL_EPOCHS=25000
    EPOCH_LAYERS_ADDED=32500
    TEST_CASE="8. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with all neurons unlocked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":1,"layers_neurons":[14, 6, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[false,false,false,false],"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    INITIAL_NETWORK="${10}"
    LAYERS="4,13,4"
    NEW_LAYERS="4,14,6,4"
    S_LAYERS=`echo ${LAYERS} | sed "s/,/|/g"`
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    ALL_EPOCHS=25000
    EPOCH_LAYERS_ADDED=40000
    TEST_CASE="9. Starting from ${S_LAYERS} network to bigger ${S_NEW_LAYERS} with all neurons unlocked ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    added_layer_test '[{"command":"modify_layers","position":1,"layers_neurons":[14, 6, 4],"conditional":true,"condition_epoch>":EPOCH},{"command":"set_layers_locks","locks":[false,false,false,false],"conditional":true,"condition_epoch>":EPOCH}]' "${TEST_CASE}" "${LAYERS}" "${EPOCH_LAYERS_ADDED}"

    clean_command_file
    NEW_LAYERS="4,6,7,7,4"
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    ALL_EPOCHS=50000
    init_network "${NEW_LAYERS}"
    TEST_CASE="10. Starting from ${S_NEW_LAYERS} network without change, an upper bound ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    static_network_test "${TEST_CASE}" "${NEW_LAYERS}"

    clean_command_file
    NEW_LAYERS="4,14,6,4"
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    ALL_EPOCHS=50000
    init_network "${NEW_LAYERS}"
    TEST_CASE="11. Starting from ${S_NEW_LAYERS} network without change, an upper bound ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    static_network_test "${TEST_CASE}" "${NEW_LAYERS}"

    clean_command_file
    NEW_LAYERS="4,13,4"
    S_NEW_LAYERS=`echo ${NEW_LAYERS} | sed "s/,/|/g"`
    ALL_EPOCHS=50000
    init_network "${NEW_LAYERS}"
    TEST_CASE="12. Starting from ${S_NEW_LAYERS} network without change, a lower bound ${RUN}"
    LOG_NAME=`echo ${TEST_CASE} | sed "s/ /_/g"`
    LOG="/tmp/results_${LOG_NAME}.txt"
    static_network_test "${TEST_CASE}" "${NEW_LAYERS}"

done
