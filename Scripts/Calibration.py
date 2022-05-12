import Objects.Calibrator as Calibrator

calibrator = Calibrator.Calibrator()
calibrator.calibrate(use_rectify=True)
calibrator.exportCalibration()
