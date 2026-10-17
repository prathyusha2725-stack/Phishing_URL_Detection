# Phishing URL Detection

A machine learning based web application that detects whether a given URL is **Phishing** or **Legitimate**.

The project uses the **PhiUSIIL Phishing URL Dataset** for training and a machine learning model to classify URLs. A Flask web application provides a simple interface where users can enter a URL and get a prediction.

## Project Overview

Phishing attacks are a common cybersecurity threat where attackers create fake websites or URLs to steal sensitive information such as usernames, passwords, banking details, and personal information.

This project aims to identify potentially phishing URLs using machine learning techniques.

The trained model analyzes URL patterns and predicts whether the URL is:

- **Phishing**
- **Legitimate**

## Features

- Machine learning based URL classification
- Flask web interface
- URL prediction through a simple input form
- Trained using the PhiUSIIL Phishing URL Dataset
- Model saved using Joblib
- High classification accuracy on the test dataset
- Easy to run locally

## Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Joblib
- HTML
- CSS
- Git & GitHub

## Dataset

This project uses the **PhiUSIIL Phishing URL Dataset**.

The dataset contains URLs classified into phishing and legitimate categories and is used to train the machine learning model.

Dataset file:

```text
dataset/PhiUSIIL_Phishing_URL_Dataset.csv
