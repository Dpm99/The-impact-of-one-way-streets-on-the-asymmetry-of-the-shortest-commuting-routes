# The-impact-of-one-way-streets-on-the-asymmetry-of-the-shortest-commuting-routes

This repository contains all the important code and data used in the paper: "The impact of one-way streets on the asymmetry of the shortest commuting routes"

### If you any of the code in this repository please cite the above paper

## Structure:

As per the above paper we conducted the specificed studies in 10 different cities using the OSMnx library. Since OSMnx uses data from OpenStreetMaps which is constantly being update, we decided to keep a specific version of the data (kept in the Network_Data folder). It is important to note that there are two python files for each city: one where we perform the necessary calculations with the regular city network (files kept in Not_Shuffled_City) and another where we shuffle this network (files kept in Shuffled_city). Finally the Analysis folder contains the code developed to perform post processing of the data and the analysis performed.

- Network_Data
- Not_Shuffled_city
- Shuffled_city
- Analysis
