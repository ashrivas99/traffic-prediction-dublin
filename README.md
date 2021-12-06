# traffic-prediction-dublin

In this project we use real world traffic data at different sites in dublin to predict the volumes of cars at a particular timeframe
```bash
#create a virtual environment
$virtualenv .env

#activate the virtual environment
$source .env/bin/activate

# install the project requirements
$pip install -r requirements.txt

#run the project
# This will save all the plots used for data-visualisation in the Plots folder
$python trafficplots.py


#This is the main file containing all the models we used
$python timeSeriesFeatures.py
```
