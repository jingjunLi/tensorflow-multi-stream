#!/bin/bash

rm noscope.pb.cc noscope.pb.h

protoc -I=. --cpp_out=.  noscope.proto




