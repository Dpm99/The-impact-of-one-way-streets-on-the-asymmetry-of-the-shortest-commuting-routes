# The-impact-of-one-way-streets-on-the-asymmetry-of-the-shortest-commuting-routes

This repository contains all the important code and data used in the paper: "The impact of one-way streets on the asymmetry of the shortest commuting routes"

### If you any of the code in this repository please cite the above paper

## Structure:

As per the above paper we conducted the specificed studies in 10 different cities using the OSMnx library. Since OSMnx uses data from OpenStreetMaps which is constantly being update, we decided to keep a specific version of the data (kept in the Network_Data folder).

- Analysis (code for the important analysis performed)
- Network Data (json files containing the original city network data used)
- Not_Shuffled_city (optimal path calculation code for each city network)
- One_way_fractions_no_shuffle (code for the calculation of the fraction of one way streets in each path for the normal city networks)
- One_way_fractions_shuffle (code for the calculation of the fraction of one way streets in each path for the shuffled city networks)
- Shuffled_city (optimal path calculation code for each shuffled city network)
