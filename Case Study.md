# Internship Case Study - December 2024

## Introduction

Welcome to our Case Study! You've made it to the final phase of our selection process, and this is your opportunity to showcase your expertise and analytical prowess. We are excited to see how you approach the tasks and the insights you can generate from the data provided. We wish you the best of luck and hope you find the challenge engaging!

## Assignment Overview

**Problem Domain:** Automotive e-commerce analytics

**Submission Deadline:** 03/12/2024 at 12:00 p.m. CET

**Submission Instructions:**
Please submit your work by emailing us your code and results in Notebook (.ipynb) and HTML formats with printed results. To ensure we receive your submission without issues, we recommend using a file-sharing service such as Google Drive or Dropbox. But preferably if you have a GitHub account, you may also create a private repository and invite us to access it. @SunguKang, @Yakal

## Context

In the rapidly evolving automotive industry, online platforms have become pivotal in driving sales and customer engagement. For this case study, you will analyze data from different car sales platforms operated by the same car company in a single market.

## Objectives

Your task is to analyze the datasets provided and extract actionable insights that can help the company optimize its sales strategy across different platforms.

### Task 1: Data Analysis

Using the provided datasets, please perform the following analyses:

1. Describe the datasets and the eventual anomalies you find. ☑️
2. Which patterns do you find in the purchasing behavior of the customers?
3. Conduct a thorough exploratory data analysis to understand the datasets. ☑️
4. Perform customer segmentation to cluster users based on their interactions and purchasing behaviors on the different platforms.
5. Investigate the types of cars (e.g., electric, hybrid, diesel) preferred on each platform and how this correlates with the platform's sales performance and user satisfaction. ☑️
6. (optional) Open-ended exploration: you can explore the datasets further and propose additional analyses, modeling, visualizations, or insights. ☑️

### Task 2: Presentation

You are the lead eCommerce Data Scientist in a well-known car company, and this task requires you to come up with ideas for the following digitalization problem: `<br>`
Online sales have dominated the used car market for years, and it makes sense for this trend also to invade the new car scene. Consequently, your company wants to increase the share of its online sales from its total new car sales as well.

Now, it is your job to develop ideas that revolve around using big data and data science within eCommerce to boost online sales.

1. Use your findings from Task 1 to provide strategic recommendations for each platform, focusing on aspects such as user experience and marketing strategies.
2. Develop ideas that focus on data science and machine learning to enhance the digital customer journey on our website and how would you reduce unnesarry user journey.

Prepare a 20-25 minute presentation of your findings, dedicating  ~20 minutes to Task 1 and ~5 minutes for Task 2.

Feel free to use any presentation software you are comfortable with.

### Considerations

- You will need to formulate hypotheses and assumptions to complete this task.
- There is no single "correct" solution; we are looking for your unique analytical approach.
- Your ability to derive business insights from the analysis is crucial.
- For task 1, you can present it with your result / HTML file or code, and for task 2, you can present it using presentation like powerpoint.

## Dataset Descriptions (Metadata)

Below are descriptions of the datasets you will be working with:

#### Users Table (users.csv)

- `customer_id`: Unique identifier for the user.
- `user_first_name`: First name of the user.
- `user_last_name`: Last name of the user.
- `gender`: Gender of the user.
- `email`: Email address of the user.

#### Cars Table (cars.csv)

- `car_id`: Unique identifier for the car.
- `car_model`: Model of the car.
- `fuel_type`: Fuel type of the car.
- `release_date`: Date when the car was released.
- `price`: Base price of the car.

#### Sales Table (sales.csv)

- `transaction_id`: Unique identifier for the purchase.
- `customer_id`: Identifier for the user who made the purchase.
- `car_id`: Identifier for the car that was purchased.
- `platform`: Platform on which the purchase was made.
- `purchase_date`: Date when the purchase was made.
- `purchase_price`: Final price at which the car was sold after any discounts.
- `user_review`: An optional user review (as a score) given for the platform after each purchase.

#### Visit Table (visits.csv)

- `visit_id`: Unique identifier for the visit.
- `customer_id`: Identifier for the user who made the visit.
- `start_timestamp`: Timestamp when the visit started (website entry or dealership entry).
- `end_timestamp`: Timestamp when the visit ended.
- `visit_type`: Type of visit (e.g., purchasing, car configuration, testing).
- `transaction_id`: purchase identifier for purchasing visits.

**Good luck and have fun!**
