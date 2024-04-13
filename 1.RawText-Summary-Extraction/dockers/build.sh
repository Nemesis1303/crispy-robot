#!/bin/bash
cd generic/
docker build  --tag 'python39_0' .
docker image ls | grep python39_0

