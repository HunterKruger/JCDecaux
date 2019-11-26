# JCDecaux
This repository contains the code for applying the Intern Data Scientist at JCDecaux.

The script is saved in 2 forms: .py and .ipynb

In the python script, I used KMeans to do clustering for *Brisbane_CityBike.json*

Firstly, I used 2 features, "latitude" and "longitude", to do the clustering.<br>
To choose the hyperparameter K, I used "elbow function", calinski_harabasz_score and silhouette_score to make decision.

Then, I used 1 feature, "number", to do the clustering.<br>
Althongh one-feature clustering is not so meaningful, but I found an extreme large sample.

To name the bike stations, I have a good idea:<br>
For the extreme big station, we can call it 'Super Station'<br>
For the rest, we can call them 'Normal Station'

Using Jupyter Notebook to open *ClusteringForCityBike.ipynb* is highly recommanded, because you can see the analysis and results in one file.

Install Jupyter Notebook by: 
+ *pip install jupyter notebook*

"ClusteringForCityBike.py" is witten in python version 3.6

Grant access by: 
+ *chomd 777 ClusteringForCityBike.py*

Install related packages by:
+ *pip install pandas==0.25.3*
+ *pip install numpy==1.17.3*
+ *pip install matplotlib==3.1.1*
+ *pip install scipy==1.3.1*
+ *pip install scikit-learn==0.21.3*

Complie it by: 
+ *./ClusteringForCityBike.py*

Link to GitHub:https://github.com/HunterKruger/JCDecaux
