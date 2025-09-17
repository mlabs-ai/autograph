#!/bin/bash
directory="${1:-data/test}"
for dot_file in "$directory"/*; do
  base_name="${dot_file%.dot}"
  svg_file="${base_name}.svg"
  dot -Tsvg "$dot_file" -o "$svg_file"
done