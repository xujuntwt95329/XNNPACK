#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## Unit tests #################################
tools/generate-pad-test.py --spec test/xx-pad.yaml --output test/xx-pad.cc &

wait
