#!/bin/bash

for folder in $(ls); do
    for file in $(ls $folder); do
        sudo convert ./$folder/$file -resize 299x299\! ./$folder/$file;
     done;
 done;
