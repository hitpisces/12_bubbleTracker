# -*- coding: utf-8 -*-
# @Time    : 20/2/2 17:07
# @Author  : Jay Lam
# @File    : testCUDA.py
# @Software: PyCharm


import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests


class cuda_test(NewOpenCVTests):
    def setUp(self):
        super(cuda_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    def test_cuda_upload_download(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = cv.cuda_GpuMat()
        cuMat.upload(npMat)

        self.assertTrue(np.allclose(cuMat.download(), npMat))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
