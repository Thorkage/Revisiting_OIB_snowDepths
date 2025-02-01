# Revisiting Operation IceBridge snow-on-sea-ice measurements in the Arctic ✈️❄️

This repository represents the research I did during my master thesis project (the thesis itself is contained in /Documentation). The aim was to highlight strenghts and weaknesses of the NASA Operation IceBridge (OIB) snow depth measurements performed with the snow radar by validating with in-situ data.

### Methodology
The comparison is based on gridded laser scanner and in-situ snow depth measurements which are then compared with the snow radar estimate of the air--snow and snow--ice interface. Since there are multiple algorithms to retrieve the interfaces (and thus snow depth), these are further compared to the official NSIDC data product. pySnowRadar (https://github.com/kingjml/pySnowRadar) is utilized to implement the CWT (Newman et al., 2014) and PEAK (Jutila et al., 2022) algorithms.

### Necessary data
To reproduce the results (or do own experimentation), there are several necessary datasets. For most of them I included a downloading script in /Downloading, which loads the data into /Data.

From OIB the following data products are used :
1. OIB L1B snow radar echograms (https://nsidc.org/data/irsno1b/versions/2)
2. OIB L1B ATM Elevation (https://nsidc.org/data/ilatm1b/versions/2)
3. OIB L4 Quicklook data product (https://nsidc.org/data/nsidc-0708/versions/1)
A (free) NASA earthdata account is necessary to get the data.

The in-situ measurements can be found on Josh Kings GitHub:
1. ECCC Eureka 2014 campaign (https://github.com/kingjml/ECCC-Eureka-2014-Snow-on-Sea-Ice-Campaign)
2. ECCC Eureka 2016 campaign (https://github.com/kingjml/ECCC-Eureka-2016-OpenData)

For large scale analysis the University of Bremens multi-year ice concentration data is used: https://seaice.uni-bremen.de/ice-type/

### References

Newman, Thomas, Sinead L. Farrell, Jacqueline Richter-Menge, Laurence N. Connor, Nathan T. Kurtz, Bruce C. Elder, and David McAdoo. “Assessment of Radar-Derived Snow Depth over Arctic Sea Ice.” Journal of Geophysical Research: Oceans 119, no. 12 (2014): 8578–8602. https://doi.org/10.1002/2014JC010284.

Jutila, Arttu, Joshua King, John Paden, Robert Ricker, Stefan Hendricks, Chris Polashenski, Veit Helm, Tobias Binder, and Christian Haas. “High-Resolution Snow Depth on Arctic Sea Ice From Low-Altitude Airborne Microwave Radar Data.” IEEE Transactions on Geoscience and Remote Sensing 60 (2022): 1–16. https://doi.org/10.1109/TGRS.2021.3063756.

