#!/bin/bash
i=1
while [ $i -le 40 ]
   do
       echo $i
       python3 train.py
       i=`expr $i + 1`
   done
