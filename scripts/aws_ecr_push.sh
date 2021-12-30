#/bin/bash

ACCOUNT_ID=720313667338
REGION=ap-southeast-1
IMAGE_NAME=reader_service
REPOSITORY_NAME=reader_service

aws ecr get-login-password | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME

docker tag $IMAGE_NAME $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME