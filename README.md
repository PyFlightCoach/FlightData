## FlightData
This repo is contains a set of datastructures and tools for handling flight log data.

### Flight 
The Flight object represents the data logged by a flight controller. The class wraps a pandas dataframe which is indexed on a single time axis. Where data is logged at different rates for different sensors it is mapped to the closest time index. Attribute access provides individual columns or sets of columns in the groups defined in Fields. Item access subsets the data in the time axis. 

### Table
The Table is the base type for most of the datastructures. It allows attribute access to individual columns. Attribute access is also available to return basic entities subclassed from the base type in the pfc-geometry package. For example in the state object table.x provides the x position, table.pos provides a Point representing the xyz position. columns that are not represented by geometric base types are considered to be labels for the data.

### State
The State object is a table representing the position and orientation of the aircraft along with their derivatives, it can be constructed from a Flight or from scratch by extrapolating in lines or around arcs. Many tools are provided to manipulate the data. The position and attitude are in a reference frame (with Z up), the derivatives move with the aircraft in either the body, wind, stability or track (like the wind axis but with no wind) frame.  


Further documentation will be provided here: https://pfcdocumentation.readthedocs.io/pyflightcoach/flightdata.html


### Installation

```bash
pip install pfc-geometry
```

### Setup from source

```bash
pip install .
```