AI-Driven Real-Time Mule Account Detection & UPI Fraud Prevention System
📌 Project Overview

Digital payment systems like UPI have made transactions seamless, but they have also created opportunities for money mule networks and coordinated fraud. This project presents an AI-powered fraud detection system designed to identify suspicious accounts and transactions before fraud is completed.

The system focuses on behavioral intelligence + network analysis to detect patterns that differentiate genuine users from mule accounts. Instead of only reacting after fraud occurs, this solution demonstrates a proactive, pre-authorization risk evaluation layer.

🎯 Objective

The goal of this project is to build a real-time risk assessment framework that:

Detects money mule accounts

Identifies abnormal transaction behavior

Understands relationships between accounts

Prevents fraudulent transfers before completion

Provides explainable reasons for every decision

Core Concepts Used

This system combines multiple intelligent techniques:

Behavioral Analysis

The model studies user activity patterns such as:

High transaction velocity

Rapid fund forwarding

Sudden spikes in activity

Interactions with many unrelated accounts

These signals are strong indicators of mule behavior.

Network-Based Detection

Fraud often happens in coordinated groups, not isolated accounts.
This project uses network relationships to:

Identify suspicious clusters

Detect accounts acting as bridges in fund movement

Expose mule chains and layered transfers

Risk-Based Pre-Authorization

Before a transaction is completed, the system:

Evaluates transaction risk

Assigns a risk score

Triggers protective action if needed

Possible actions:

OTP verification

Transaction hold

Alert generation

Explainable AI (XAI)

Every flagged transaction includes clear, human-readable reasons such as:

“Unusual transaction burst”
“Funds forwarded within seconds”
“Connected to high-risk account cluster”

This makes the system usable for fraud investigators and compliance teams.
Project Structure
├── app.py                  # Main application logic
├── model/                  # Trained ML models & policies
├── data/                   # Engineered feature datasets
├── raw_data/               # Simulated transaction data
├── graphs/                 # Network graph examples
├── lib/                    # Visualization libraries
├── train_model.ipynb       # Model training pipeline
├── feature_engineering.ipynb
├── simulate_data.ipynb
└── Scoring.py              # Risk scoring logic

How the System Works (Flow)

Transaction occurs

Features are extracted from user behavior

Network relationship signals are added

AI model predicts fraud risk

Risk policy engine evaluates severity

System decides:

Allow

Verify

Block / Hold

Technologies & Skills Demonstrated

Machine Learning for fraud detection

Behavioral feature engineering

Network / graph-based fraud analysis

Risk scoring systems

Explainable AI concepts

Python-based model deployment logic

Dataset

This project uses simulated transaction and account data to model real-world fraud patterns safely. The dataset includes:

Account labels

Transaction records

Simulated mule behavior

Normal user patterns

Real-World Impact

This system demonstrates how AI can be used to:

Reduce financial fraud losses

Detect mule networks early

Protect digital payment ecosystems

Assist fraud investigation teams

It showcases a practical fintech security solution rather than just a theoretical ML model.

Author

Sai Sugeet
Engineering Student | AI & Security Enthusiast
