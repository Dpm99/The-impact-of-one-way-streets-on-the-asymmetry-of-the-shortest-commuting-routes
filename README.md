# The-impact-of-one-way-streets-on-the-asymmetry-of-the-shortest-commuting-routes

This repository contains all the important code and data used in the paper: [Impact of one-way streets on the asymmetry of the shortest commuting routes](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.023053)

### If you use any of the code in this repository please cite the above paper

## Structure:

As per the above paper we conducted the specificed studies in 10 different cities using the OSMnx library. Since OSMnx uses data from OpenStreetMaps which is constantly being update, we decided to keep a specific version of the data (unable to upload this because it's larger than what GitHub allows, therefore you can either ask for the JSON's we used or download the most recent versions from OpenStreetMaps).

- Analysis (code for the important analysis performed)
- Not_Shuffled_city (optimal path calculation code for each city network)
- One_way_fractions_no_shuffle (code for the calculation of the fraction of one way streets in each path for the normal city networks)
- One_way_fractions_shuffle (code for the calculation of the fraction of one way streets in each path for the shuffled city networks)
- Shuffled_city (optimal path calculation code for each shuffled city network)
