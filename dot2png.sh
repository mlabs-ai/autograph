#!/bin/bash
for dot_file in data/test/*.dot; do
  base_name="${dot_file%.dot}"
  png_file="${base_name}.png"
  dot -Tpng "$dot_file" -o "$png_file"
done