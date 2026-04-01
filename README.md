# Pocket Professor

## Description
**Pocket Professor** is a compact (~38-centimeter) physical Information Technology assistant built using a **Raspberry Pi**, **open-source AI**, and **voice interaction**.

The device observes a user’s workspace through a camera, listens to spoken IT-related questions, and responds with **hint-based verbal guidance** rather than full solutions.

The goal is to provide **hands-free troubleshooting support** while keeping users focused and free from distractions like phones or web browsing.

##

## Table of Contents
- [Project Team](#projectteam)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Scope](#project-scope)
- [Acceptance Criteria](#acceptance-criteria)
- [Repository Structure](#repository-structure)
- [Credits](#credits)
- [License](#license)

##

## Project Team
- Saige Gee
- Lukas Gairdner
- Bekah Vanderzee

##

## Installation

### Hardware Requirements
- Raspberry Pi 5  
- Raspberry Pi Camera 3  
- USB microphone  
- USB speaker  
- 18650 rechargeable battery system  
- 3D-printed housing  

### Software Requirements
- Python 3.11
- Speech-to-Text library  
- Text-to-Speech library  
- Open-source AI model compatible with Raspberry Pi  
- Raspberry Pi camera and audio drivers  

### Setup Steps
1. Assemble Raspberry Pi with camera, microphone, and speaker. 
2. Install Raspberry Pi OS and required drivers.  
3. Install Python dependencies and the AI model.  
4. Configure speech-to-text and text-to-speech.  
5. Place hardware inside the 3D-printed Pocket Professor housing.  
6. Glue down Velcro strips to hold the 3D-printed Pocket Professor housing together.
7. Power on and test interaction.

##

## AI Model Setup

The AI model file is too large to store in this repository.

Download the three zipped files from the code/models folder

After downloading, create a `models/` folder in the project directory and unzip the files inside it.


##

## Code Setup

### Requirements
- Python 3.11 or higher
- USB microphone
- USB speaker
- Pi camera 

### Installation

1. Clone the repository
2. Install the dependencies globally, not in any venv
   ```
   python3.11 -m pip install -r requirements.txt
   ```
3. Download the AI model (see AI Model Setup section above)

4. Run the program:
   python pocket_professor.py

##

## Usage
1. Turn on the Pocket Professor.
2. State wake-word
3. Ask an **IT-related** question using your voice.  
4. Show relevant workspace items to the camera if needed.  
5. Receive a **spoken hint** to guide troubleshooting.  

##

## Features
- Reliable **power management** with safe shutdown  
- **Live camera perception** of workspace  
- **Hands-free voice interaction**  
- **Hint-based AI responses** (no full answers)  
- **3D-printed professor housing** that sits upright 
- Fully **local, open-source AI processing**

##

## Project Scope

### In Scope
- Physical 8–10 inch professor model  
- Raspberry Pi with open-source AI  
- Camera-based visual understanding  
- Voice input and output  
- Stand-alone IT hint assistant  

### Out of Scope
- Mobile or web companion app  
- Non-English language support  
- Complex physical animations  
- Voice training or personalization  

##

## Acceptance Criteria
The Pocket Professor is considered complete when it:

- Powers on and off reliably  
- Sits Upright without support
- Responds to spoken IT questions  
- Provides relevant hints within **5 seconds**    
- Maintains consistent and appropriate responses  
