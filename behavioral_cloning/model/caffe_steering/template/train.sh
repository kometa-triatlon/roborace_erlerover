#!/bin/bash
mkdir -p log
mkdir -p train
caffe train --solver solver.prototxt -log_dir log
