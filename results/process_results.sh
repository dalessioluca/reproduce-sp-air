#!/bin/sh

cat EXTRAPOLATION_WITH_GRID/with_grid.csv | grep test_in | awk -F',' '{print $3,$2}' > processed/with_grid_test.txt
cat EXTRAPOLATION_WITH_GRID/with_grid.csv | grep test_out | awk -F',' '{print $3,$2}' > processed/with_grid_extrapolation.txt

cat EXTRAPOLATION_NO_GRID/no_grid.csv | grep test_in | awk -F',' '{print $3,$2}' > processed/no_grid_test.txt
cat EXTRAPOLATION_NO_GRID/no_grid.csv | grep test_out | awk -F',' '{print $3,$2}' > processed/no_grid_extrapolation.txt

cat EXTRAPOLATION_NO_GRID_V2/no_grid_v2.csv | grep test_in | awk -F',' '{print $3,$2}' > processed/no_grid_v2_test.txt
cat EXTRAPOLATION_NO_GRID_V2/no_grid_v2.csv | grep test_out | awk -F',' '{print $3,$2}' > processed/no_grid_v2_extrapolation.txt


