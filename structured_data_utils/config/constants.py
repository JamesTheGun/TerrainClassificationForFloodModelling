ESPSG = "EPSG:7856" #we are just going to assume everything is in aus...
RES = "0.5"

#you might be asking, "why the fuck do we have
#seperate standardised files if we make the geotifs anyway?"
#Its because we will want to run on geotiffs we have not made
#going forwards, and this keeps the pipeline consistant
GEOTIFF_LOCATIONS_TO_CORRESPONDING_STANDARDISED_LOCATION = {
    "data/POSITIVE.tif":"data/POSITIVE_STANDARDISED.tif",
    "data/NEGATIVE.tif":"data/NEGATIVE_STANDARDISED.tif",
    "data/COMBINED.tif":"data/COMBINED_STANDARDISED.tif",
    "data/TEST_TIFF.tif":"data/COMBINED_TEST_STANDARDISED.tif",
}

EMPTY_VAL = -9999