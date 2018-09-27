#!/bin/bash
i=1
while [ $i -le 20 ]
   do
       python train.py
       i=`expr $i + 1`
   done
