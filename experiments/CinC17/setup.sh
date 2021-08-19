#!/bin/bash

url=https://www.physionet.org/content/challenge-2017/1.0.0/

cd experiments/CinC17
mkdir data && cd data

curl -O $url/training2017.zip
unzip training2017.zip
curl -O $url/sample2017.zip
unzip sample2017.zip
curl -O $url/REFERENCE-v3.csv

