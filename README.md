# A Two-Stage Approach to Routing with Driver Preferences via Heatmaps

This github includes code to both build and run vehicle routes to match high quality driver routes. The code is broken into two files:

-- `model_build.py`: Contains code to construct heatmaps from historical driver routes
-- `model_apply.py`: Contains code to run new routes.

Both files use paths based on the file structure of the MIT-Amazon Last Mile Vehicle Routing challenge. Update the file paths at the beginning of each file to use within a new code structure. 

### Dependencies
Our code is built upon pandas, ortools, numpy, and pebble (version 4.6.1.)
