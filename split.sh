#!/bin/bash

N=$1
shift

## Grab data from standard input
cat | gshuf > temp.$$.txt

## All but last N
ghead -n -${N} temp.$$.txt > temp.$$.train.txt

## Only last N
gtail -n ${N} temp.$$.txt > temp.$$.test.txt

## Run command with split data appended
$@ temp.$$.train.txt temp.$$.test.txt

## Remove temporary files
rm temp.$$.*
