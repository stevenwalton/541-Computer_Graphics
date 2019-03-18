# Load Data
#OpenDatabase("/research/earthquake/sw4_out.cycle_001100.root", 0)
OpenDatabase("/research/earthquake/sw4_out.cycle_*.root database", 0)

# Add Plot
AddPlot("Pseudocolor", "mesh_topo/something_magnitude", 1, 1)
AddOperator("Threshold", 1)
ThresholdAtts = ThresholdAttributes()
ThresholdAtts.outputMeshType = 0
ThresholdAtts.boundsInputType = 0
ThresholdAtts.listedVarNames = ("default")
ThresholdAtts.zonePortions = (1)
ThresholdAtts.lowerBounds = (0.02)
ThresholdAtts.upperBounds = (1e+37)
ThresholdAtts.defaultVarName = "mesh_topo/something_magnitude"
ThresholdAtts.defaultVarIsScalar = 1
ThresholdAtts.boundsRange = ("0.02:1e+37")
SetOperatorOptions(ThresholdAtts, 1)
# Pseudo Color Attributes
PseudocolorAtts = PseudocolorAttributes()
PseudocolorAtts.scaling = PseudocolorAtts.Linear  # Linear, Log, Skew
PseudocolorAtts.skewFactor = 1
PseudocolorAtts.limitsMode = PseudocolorAtts.OriginalData  # OriginalData, CurrentPlot
PseudocolorAtts.minFlag = 0
PseudocolorAtts.min = 0
PseudocolorAtts.maxFlag = 1
PseudocolorAtts.max = 2
PseudocolorAtts.centering = PseudocolorAtts.Natural  # Natural, Nodal, Zonal
PseudocolorAtts.colorTableName = "hot"
PseudocolorAtts.invertColorTable = 0
PseudocolorAtts.opacityType = PseudocolorAtts.FullyOpaque  # ColorTable, FullyOpaque, Constant, Ramp, VariableRange
PseudocolorAtts.opacityVariable = ""
PseudocolorAtts.opacity = 1
PseudocolorAtts.opacityVarMin = 0
PseudocolorAtts.opacityVarMax = 1
PseudocolorAtts.opacityVarMinFlag = 0
PseudocolorAtts.opacityVarMaxFlag = 0
PseudocolorAtts.pointSize = 0.05
PseudocolorAtts.pointType = PseudocolorAtts.Point  # Box, Axis, Icosahedron, Octahedron, Tetrahedron, SphereGeometry, Point, Sphere
PseudocolorAtts.pointSizeVarEnabled = 0
PseudocolorAtts.pointSizeVar = "default"
PseudocolorAtts.pointSizePixels = 2
PseudocolorAtts.lineStyle = PseudocolorAtts.SOLID  # SOLID, DASH, DOT, DOTDASH
PseudocolorAtts.lineType = PseudocolorAtts.Line  # Line, Tube, Ribbon
PseudocolorAtts.lineWidth = 0
PseudocolorAtts.tubeResolution = 10
PseudocolorAtts.tubeRadiusSizeType = PseudocolorAtts.FractionOfBBox  # Absolute, FractionOfBBox
PseudocolorAtts.tubeRadiusAbsolute = 0.125
PseudocolorAtts.tubeRadiusBBox = 0.005
PseudocolorAtts.tubeRadiusVarEnabled = 0
PseudocolorAtts.tubeRadiusVar = ""
PseudocolorAtts.tubeRadiusVarRatio = 10
PseudocolorAtts.tailStyle = PseudocolorAtts.None  # None, Spheres, Cones
PseudocolorAtts.headStyle = PseudocolorAtts.None  # None, Spheres, Cones
PseudocolorAtts.endPointRadiusSizeType = PseudocolorAtts.FractionOfBBox  # Absolute, FractionOfBBox
PseudocolorAtts.endPointRadiusAbsolute = 0.125
PseudocolorAtts.endPointRadiusBBox = 0.05
PseudocolorAtts.endPointResolution = 10
PseudocolorAtts.endPointRatio = 5
PseudocolorAtts.endPointRadiusVarEnabled = 0
PseudocolorAtts.endPointRadiusVar = ""
PseudocolorAtts.endPointRadiusVarRatio = 10
PseudocolorAtts.renderSurfaces = 1
PseudocolorAtts.renderWireframe = 0
PseudocolorAtts.renderPoints = 0
PseudocolorAtts.smoothingLevel = 0
PseudocolorAtts.legendFlag = 1
PseudocolorAtts.lightingFlag = 1
PseudocolorAtts.wireframeColor = (0, 0, 0, 0)
PseudocolorAtts.pointColor = (0, 0, 0, 0)
#
## Threshold Attributes
#ThresholdAtts = ThresholdAttributes()
#ThresholdAtts.outputMeshType = 0
#ThresholdAtts.boundsInputType = 0
#ThresholdAtts.listedVarNames = ("default")
#ThresholdAtts.zonePortions = (1)
#ThresholdAtts.lowerBounds = (0.002)
#ThresholdAtts.upperBounds = (1e+37)
#ThresholdAtts.defaultVarName = "mesh_topo/something_magnitude"
#ThresholdAtts.defaultVarIsScalar = 1
#ThresholdAtts.boundsRange = ("0.02:1e+37")
#
## Set Options
#SetOperatorOptions(ThresholdAtts, 1)
#SetPlotOptions(PseudocolorAtts)
#
#DrawPlots()


# Test 3D view
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

swatts = SaveWindowAttributes()
swatts.family = 0
swatts.format = swatts.PNG
swatts.width = 1024
swatts.height = 1024
file_index = 0
#swatts.fileName = "test.png"
#SetSaveWindowAttributes(swatts)
#SaveWindow()
nts = TimeSliderGetNStates()
base_fName = "/home/users/swalton2/Pictures/slicing/frame_"
# Steping
for ts in range(0, nts):
    TimeSliderSetState(ts)
    DrawPlots()
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
