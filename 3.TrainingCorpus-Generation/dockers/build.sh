#!/bin/bash
docker build  --tag 'python39_3' .
docker image ls | grep python39_3
