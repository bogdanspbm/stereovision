import Objects.Camera as Camera

camera = Camera.VirtualCamera(0, 3840, 1080)

camera.showVideo(filter="CORNERS", filter_params=[(9, 14)])
