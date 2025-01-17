# Ice Cream Recommendation System

A simple and user-friendly **Ice Cream Recommendation System** built using **Google Sheets**, **Google Forms**, and **Streamlit**. This project helps users discover their ideal ice cream flavor based on their preferences and inputs.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Workflow](#workflow)
- [License](#license)

---

## Overview
The Ice Cream Recommendation System collects user inputs through a **Google Form** and recommends the best-matching ice cream flavor based on predefined criteria. The backend logic processes the collected data and displays the result on a **Streamlit** app interface.The system takes user inputs, such as age, favorite flavor, toppings, and zodiac sign, to provide personalized recommendations. Designed for both fun and practical use, it showcases how data-driven decisions can enhance user experience

---

## Features
- **Interactive Google Form**: Users can input their preferences, such as:
  - Favorite flavor profiles (e.g: chocolate, vanilla, strawberry)
  - Favourite toppings (Nuts, Chocos, etc)
  - Gender
  - Age
- **Dynamic Recommendations**: Provides personalized ice cream suggestions based on the collected data.
- **Real-time Processing**: Automatically updates and processes user responses via Google Sheets.
- **Streamlit Frontend**: Displays the recommendations with an intuitive and clean UI.

---

## Tech Stack
- **Google Sheets**: Backend data storage and processing.
- **Google Forms**: Frontend data collection from users.
- **Streamlit**: Web-based Python framework for displaying recommendations.

---

## Workflow
1. **Data Collection**: Users fill out a Google Form with their preferences.
2. **Data Storage**: Responses are automatically saved to a linked Google Sheet.
3. **Processing Logic**: The backend logic (Python script) processes the data from Google Sheets to compute the best recommendations.
4. **Recommendation Display**: Results are shown on a Streamlit web app in an interactive manner.
