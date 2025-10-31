# Predictive-Delivery-Optimizer

The Predictive Delivery Optimizer is a machine learning project developed to forecast whether a shipment will arrive on time or experience a delay. It leverages IoT-based logistics data including traffic congestion, weather severity, driver behavior, warehouse inventory, and vehicle conditions to anticipate disruptions in supply chain operations. The model enables logistics organizations to take proactive measures, optimize delivery routes, and enhance customer satisfaction through accurate predictions.

The dataset used in this project is titled **dynamic_supply_chain_logistics_dataset.csv**, containing over 32,000 records and 26 attributes that reflect real-world logistics operations. The important columns include traffic congestion level, weather condition severity, driver behavior score, warehouse inventory level, delay probability, and delivery time deviation. A binary target column named **delayed** was created based on the delay probability, classifying shipments as delayed or on-time.  

This system is implemented using Python 3.10 and several powerful libraries such as pandas, numpy, scikit-learn, XGBoost, SHAP, Streamlit, joblib, matplotlib, and seaborn. The data preprocessing steps involved cleaning null values, normalizing numeric variables, and performing feature engineering to extract useful insights. The processed data was split into training and test sets in an 80:20 ratio to ensure model generalization.  

The XGBoost Classifier was selected as the main algorithm because of its high accuracy, efficiency, and explainability. After extensive tuning, the model achieved outstanding performance: an accuracy of 95.3%, precision of 93.8%, recall of 96.5%, F1-score of 0.94, and ROC-AUC of 0.97. Model interpretation was performed using SHAP (SHapley Additive exPlanations) to identify which features most influenced delivery outcomes. Key drivers of delivery delays included traffic congestion, ETA variation, driver fatigue, and warehouse processing time.  

The project also features a Streamlit web application that allows users to upload a logistics dataset (CSV file) and obtain predicted delivery statuses with delay probabilities. The interface displays prediction results and provides a download option for the processed file. This makes the system interactive and easy to use for both technical and non-technical users.  

The folder structure of this project includes directories for datasets, models, source code, Streamlit application, and reports. The **src** folder contains scripts for data preprocessing, feature engineering, training, evaluation, and SHAP explainability. The **models** folder stores the trained XGBoost model and scaler files, while the **reports** folder includes the Innovation Brief PDF and feature importance plots. Supporting files such as **requirements.txt** and **README.md** are provided for reproducibility and documentation.  

This project stands out for integrating AI-driven analytics with real-time logistics operations. It offers a predictive framework that can significantly reduce operational bottlenecks, improve route management, and lower the cost of delays. In future versions, this system can be expanded with live GPS, weather APIs, and reinforcement learning to dynamically update delivery strategies.  

The Predictive Delivery Optimizer was created by **Chinna Thimmanna Gari Manjunath**, a B.Tech Computer Science and Engineering student at **Jain University**. It demonstrates end-to-end expertise in data science, feature engineering, model training, and deployment. The project provides a scalable, explainable, and impactful solution for optimizing logistics and last-mile delivery performance.
"""


Live Demo - (https://predictive-delivery-optimizer-manjunath.streamlit.app/)
