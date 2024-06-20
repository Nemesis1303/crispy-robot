#!/bin/bash
docker build  --tag 'python39_2' .
docker image ls | grep python39_2

