import Objects.Calibrator as Calibrator

'''
Call this script to calibrate your camera using your calibrating images
-----------------------------------------------------------------------
Save your calibrating images into 'Frames' folder before the start
-----------------------------------------------------------------------
All calibration results would be saved in a 'config' folder
'''

calibrator = Calibrator.Calibrator()
calibrator.calibrate(use_rectify=True)
calibrator.exportCalibration()
