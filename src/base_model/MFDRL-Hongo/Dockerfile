FROM ubuntu:24.04 AS build-env

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN apt-get update
RUN apt-get -y --no-install-recommends install python3.12-dev python3-pip gdal-bin libgdal-dev

WORKDIR /app

COPY requirements.txt /app/
RUN pip3 install --break-system-packages -r requirements.txt
