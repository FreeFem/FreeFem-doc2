#!/bin/sh

for file in `ls *.md`
do
	filename=$(basename -- "$file")
	extension="${filename##*.}"
	filename="${filename%.*}"
	pandoc $file -o $filename.rst
done


