# 2018 HTC Corporation. All Rights Reserved.
#
# This source code is licensed under the HTC license which can be found in the
# LICENSE file in the root directory of this work.

with open('prediction_result.csv', 'w') as f:
    f.write('Id,Category\n')
    with open('resnet50_prediction_result.csv') as ff:
        for line in ff:
            f.write(line)
    with open('mobilenetv2_prediction_result.csv') as ff:
        for line in ff:
            f.write(line)
print ('done, please submit prediction_result.csv to kaggle')
