#!/bin/bash
docker build  --tag 'python39_1' .
docker image ls | grep python39_0

