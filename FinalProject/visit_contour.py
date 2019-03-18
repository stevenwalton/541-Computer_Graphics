OpenDatabase("/research/earthquake/sw4_out.cycle_*.root database", 0)
AddPlot("Contour", "mesh_topo/something_magnitude", 1, 1)

# 3D Attributes
View3DAtts = View3DAttributes()
View3DAtts.viewNormal = (-0.7, 0.5, -0.3)
View3DAtts.focus = (50000, 50000, 15000)
View3DAtts.viewUp = (0.2, -0.2, -1)
View3DAtts.viewAngle = 30
View3DAtts.parallelScale = 74000
View3DAtts.nearPlane = -150000
View3DAtts.farPlane = 150000
View3DAtts.imagePan = (0,0)
View3DAtts.imageZoom = 1
View3DAtts.perspective = 1
View3DAtts.eyeAngle = 2
View3DAtts.centerOfRotationSet = 0
View3DAtts.centerOfRotation = (50200, 50200, 15000)
View3DAtts.axis3DScaleFlag = 0
View3DAtts.axis3DScales = (1,1,1)
View3DAtts.shear = (0,0,1)
View3DAtts.windowValid = 1

SetView3D(View3DAtts)

# Annotations
aatts = AnnotationAttributes()
aatts.axes3D.visible = 0
aatts.axes3D.triadFlag = 1
aatts.axes3D.bboxFlag = 1
aatts.userInfoFlag = 0
aatts.databaseInfoFlag = 0
aatts.legendInfoFlag = 0
SetAnnotationAttributes(aatts)

swatts = SaveWindowAttributes()
swatts.family = 0
swatts.format = swatts.PNG
swatts.width = 1024
swatts.height = 1024
file_index = 0
nts = TimeSliderGetNStates()
base_fName = "/home/users/swalton2/Pictures/contour/frame_"
for ts in range(0, nts):
    TimeSliderSetState(ts)
    time = ""
    if(ts < 10):
        time = "0" + str(ts)
    else:
        time = str(ts)
    swatts.fileName = base_fName + time + ".png"
    SetSaveWindowAttributes(swatts)
    SaveWindow()
    file_index += 1

exit()
