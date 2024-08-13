from flask import Flask, render_template, request, redirect, make_response, jsonify, send_from_directory
import json
import subprocess
from core.matsim_output import *

# app is an instance of the Flask class
app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def home():
    return render_template("index.html")

@app.route("/runSim")
def runSim():
    return render_template("runSimulation.html")


# Path to your JAR file
JAR_PATH_MATSim = 'MATSim_DMV\\out\\artifacts\\matsim_dmv_jar\\matsim-example-project.jar'
JAR_PATH_GH = 'GraphHopper_DMV\\out\\artifacts\\GraphHopper_DMV_jar\\GraphHopper_DMV.jar'


@app.route('/run-matsim')
def run_matsim():
    try:
        result = subprocess.run(['java', '-jar', JAR_PATH_MATSim], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return f"MATSim executed successfully.\nOutput:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}\nError Output:\n{e.stderr}"
    
@app.route('/run-gh')
def run_gh():
    try:
        result = subprocess.run(['java', '-jar', JAR_PATH_GH], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return f"GraphHopper executed successfully.\nOutput:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}\nError Output:\n{e.stderr}"

@app.route('/run-output')
def run_output():
    morning_traffic()
    morning_plot_all()
    morning_plot_gh()
    regression_dist()
    regression_time()
    
    return f"Outputs are saved."


@app.route("/results")
def results():
    return render_template("simulationResult.html")

@app.route("/resultStats")
def resultStats():
    mat_gh_length_triptime, vmt_mat_km, vmt_mat_mi, vmt_gh_km, vmt_gh_mi, vmt_percent_change_km, avg_trip_len_mat_km, avg_trip_len_gh_km, len_percent_change, avg_trip_time_mat_min, avg_trip_time_gh_min, time_percent_change, avg_spent_time_mat_min, avg_spent_time_gh_min, spent_percent_change = mat_gh_length_time()
    try:
        return jsonify({
            "vmt_mat_km": vmt_mat_km,
            "vmt_mat_mi": vmt_mat_mi,
            "vmt_gh_km": vmt_gh_km,
            "vmt_gh_mi": vmt_gh_mi,
            "vmt_percent_change_km": vmt_percent_change_km,
            "avg_trip_len_mat_km": avg_trip_len_mat_km,
            "avg_trip_len_gh_km": avg_trip_len_gh_km,
            "len_percent_change": len_percent_change,
            "avg_trip_time_mat_min": avg_trip_time_mat_min,
            "avg_trip_time_gh_min": avg_trip_time_gh_min,
            "time_percent_change": time_percent_change,
            "avg_spent_time_mat_min": avg_spent_time_mat_min,
            "avg_spent_time_gh_min": avg_spent_time_gh_min,
            "spent_percent_change": spent_percent_change
        })
    except FileNotFoundError as e:
        return jsonify({"error": "result stats not found."}), 404

@app.route("/morningPeak_plotAll")
def morningPeak_plotAll():   
    return send_from_directory('static/images', 'morning_traffic_all.png')

@app.route("/morningPeak_plotGH")
def morningPeak_plotGH():   
    return send_from_directory('static/images', 'morning_traffic_gh.png')

@app.route("/regression_distPlot")
def regression_distPlot():   
    return send_from_directory('static/images', 'regression_dist.png')

@app.route("/regression_timePlot")
def regression_timePlot():   
    return send_from_directory('static/images', 'regression_time.png')

if __name__ == "__main__":
    app.run(debug=True)
