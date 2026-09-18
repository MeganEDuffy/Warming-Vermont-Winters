## Divergent patterns of soil-stream connectivity during mid-winter runoff events between two catchments in the northeastern USA 

### Megan E. Duffy (megan dot duffy at uvm dot edu)

### Rubenstein School of Environment and Natural Resources, University of Vermont, Burlington, VT, USA.

### Abstract
More frequent and intense rain-on-snow events and mid-winter thermal thaws, as well as thinner and inconsistent snowpack, are occurring in the northeastern USA. Yet the effects of changing winter conditions on flowpaths and consequences for nutrient cycles and budgets are unknown. To manage and predict water quality, it is critical to understand how warming winters mobilize water and nutrients. Here, we used three complementary approaches to investigate hydrologic connectivity and nutrient mobilization during winter runoff events: stream nutrient monitoring using in situ sensors, measurement of soil nutrient availability, and event-level end-member mixing analysis (EMMA) to determine source contributions to streamflow during three separate mid-winter to spring melt events at two first-order streams – one forested and one agricultural in Vermont, USA. We found that soil-available nitrate, phosphate, and ammonium were present throughout the winter for transport.  We identified three end-member sources to streamflow: groundwater, meltwater, and soil water. The agricultural catchment, where some soils remained subfreezing all winter, displayed consistent temporal flowpath dynamics for all runoff events, with pre-event groundwater successively replaced by soil water and meltwater. In contrast, the forested catchment with consistent snowpack and above-freezing soils displayed a seasonal progression, from little apparent soil-stream connectivity during a February rain-on-snow event to increasing contributions of soil water to streamflow during a March thermal event and the spring melt. Although the EMMAs had high levels of uncertainty, together the three approaches suggest that winter in this climate is far more dynamic and relevant to downstream nutrient budgets than previously considered.

### Data:

All data used in analyses can be found in Data\
- Meteorological data from each catchment's stream outlet site in Met_data\
- Snowpack depth data from Community Collaborative Rain, Hail and Snow (CoCoRaHS) network and NOAA weather station in Met_data\
- Stream sensor (s::can spectrolyser spectrophotometer; YSI EXO2) and discharge data in StreamSensor_data\
- Soil temperature and moisture data in SoilSensor_data\
- Soil-available nutrient data from in situ ion exchange resin capsules ("soil resins") in SoilNutrient_data\
- Stream and end-member chemistry data (major cations and anions, trace metals, stable isotopes of water) in GrabSample_data\
- Integrated water and solute yields for selected winter-to-spring runoff events in EventYield_data\

### Analysis modules:

- Python module for end-member mixing analysis (EMMA) principle components analysis (PCA) and hydrograph separation steps in EMMA\
- R function to process and plot stream sensor data, soil sensor data, stream and end-member solute data, and EMMA output in Functions\
- R function to pull USGS Gauge discharge data from 1928-2025 in Functions\

### Notebooks to run all analyses

- Jupyter notebooks to apply the above modules and data are contained in Notebooks\
