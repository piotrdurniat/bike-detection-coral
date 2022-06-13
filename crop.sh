#!/usr/bin/env bash

in_file="${1}"
out_file="${2}"
center_x="${3}"
center_y="${4}"
width="${5}"

((x = center_x - width / 2))
((y = center_y - width / 2))

ffmpeg -y -i "${in_file}" -filter:v crop="${width}:${width}:${x}:${y}" -c:a copy "${out_file}"
