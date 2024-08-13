This is a demonstration of the research work on evaluating the efficacy of shortest path routing for approximating complex traffic pattern

The primary focus of this study is on assessing the viability of a fast, shortest path routing system as a method for traffic modeling. Key metrics include travel time accuracy, congestion levels, similarity in corresponding routes, Vehicle Miles Traveled, and average spending time on road. By analyzing these factors, we assess the degree to which the shortest path routing approach can approximate actual traffic conditions.

How to run the demo:

Step1:
Set up the environments:

This project runs on IntellijIdea and uses Java and Python. Make sure your system has those.

Step2:
Set up the simulation JAR files:

- The JAR files for MATSim and GraphHopper are in out.artifacts folder here (https://github.com/subrina0013/graphhopper_dmv, https://github.com/subrina0013/matsim_dmv).

- Update the path for the JAR files (matched with the location where you saved MATSim_DMV and GraphHopper_DMV folder).

- In app.py make sure the location of the JAR files are matched with the location of your JAR files.

- In matsim_output.py and gh_output.py make sure the file locations match with the saved file locations.

Step3:
Run simulations:

- Go to Run Simulations page

- Run MATSim, GraphHopper and Generate Outputs

Step4:
Visualize the results:

- Go to Simulation Results page

Following things need to be installed:

flask, subprocess, scipy, numpy, seaborn, matplotlib, re , pandas

run 'pip install -r requirements.txt' in terminal

The demo is run with FLASK_APP=app.py flask run
