ESPSG = "EPSG:7856" #we are just going to assume everything is in aus...
RES = "0.5"

#you might be asking, "why the fuck do we have
#seperate standardised files if we make the geotifs anyway?"
#Its because we will want to run on geotiffs we have not made
#going forwards, and this keeps the pipeline consistant

STANDARDISATION_TARGET_TIFFS = ["POSITIVE.tif", "COMBINED.tif"]

EMPTY_VAL = -9999