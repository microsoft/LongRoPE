#!/bin/bash

if uname -a | grep -q "GCRSANDBOX487"; then
    path_team="/data/yiran/teamdrive/ExtendSeqLen"
    #
elif uname -a | grep -q "GCRSANDBOX488"; then
    path_team="/data/yiran/teamdrive/ExtendSeqLen"
    #
elif uname -a | grep -q "GCRSANDBOX501"; then
    path_team="/data/lzhani/ExtendSeqLen/teamdrive/ExtendSeqLen"
    #
elif uname -a | grep -q "GCRSANDBOX504"; then
    path_team="/data/yiran/teamdrive/ExtendSeqLen"
    #
elif uname -a | grep -q "GCRSANDBOX506"; then
    path_team="/data/yiran/teamdrive2/ExtendSeqLen"
    #
elif uname -a | grep -q "GCRAZGDL1142"; then
    path_team="/home/v-dingyiran/teamdrive/ExtendSeqLen"
    #
elif uname -a | grep -q "GCRAZGDL1499"; then
    path_team="/scratch/yiran/teamdrive/ExtendSeqLen"
    #
elif uname -a | grep -q "aiforms2000000"; then
    path_team="/home/aisilicon/yiran/teamdrive/ExtendSeqLen"
    #
elif uname -a | grep -q "aiforms2000001"; then
    path_team="/mnt/yiran/teamdrive/ExtendSeqLen"
    #
elif uname -a | grep -q "aiforms2000002"; then
    path_team="/mnt/yiran/teamdrive/ExtendSeqLen"
    #
elif uname -a | grep -q "aiforms2000003"; then
    path_team="/mnt/yiran/teamdrive3/ExtendSeqLen"
    #
elif uname -a | grep -q "GCRAZGDL1506"; then
    path_team="/home/chengzhang/teamdrive/ExtendSeqLen"
    #
else
    echo "Neither on list"
fi
