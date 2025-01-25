import os

# Define the projects and their subfolders
projects = [
    "Project_1_Deep_Learning_xray_CTscan",
    "Project_2_Pedestrian_Light_Classification",
    "Project_3_TV_Aerial_Detection",
    "Project_4_Income_Benchmark_Prediction",
    "Project_5_Employee_Attrition_Clustering",
    "Project_6_Yelp_Sentiment_Analysis",
    "Project_7_Fault_Detection_Industrial_Systems",
    "Project_8_Spotfire_Export_To_PowerPoint",
    "Project_9_Python_Gmail_API_Automation",
    "Project_10_Custom_Spotfire_Visualizations",
    "Project_11_Python_Image_Processing_Tool",
    "Project_12_Renewable_Energy_Topic_Modeling",
    "Project_13_Sentiment_Analysis_PreTrained_Embeddings",
    "Project_14_Python_CSV_Data_Transformation",
    "Project_15_Spotfire_Accordion_Menu",
    "Project_16_Automating_Fault_Detection",
    "Project_17_Python_PowerPoint_Automation",
    "Project_18_Employee_Attrition_KMeans",
    "Project_19_Economic_Sensitivity_Analysis",
    "Project_20_Automating_Gmail_Workflows",
    "Project_21_Carbon_Emissions_Dashboard",
    "Project_22_Data_Visualization_Training"
]

subfolders = ["code", "visuals", "documentation"]

# Create folders and subfolders
for project in projects:
    try:
        # Create main project folder
        os.makedirs(project, exist_ok=True)
        
        # Create subfolders within each project
        for subfolder in subfolders:
            os.makedirs(os.path.join(project, subfolder), exist_ok=True)

    except Exception as e:
        print(f"Error creating folders for {project}: {e}")

print("Folders and subfolders created successfully!")
