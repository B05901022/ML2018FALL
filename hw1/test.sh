#!/bin/bash
i=1
while [ $i -le 20 ]
   do
       python3 train.py
       i=`expr $i + 1`
   done
