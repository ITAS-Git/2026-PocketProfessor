# Pocket Professor

## Description
**Pocket Professor** is a compact (~8-inch) physical Information Technology assistant built using a **Raspberry Pi**, **open-source AI**, and **voice interaction**.

The device observes a user’s workspace through a camera, listens to spoken IT-related questions, and responds with **hint-based verbal guidance** rather than full solutions.

The goal is to provide **hands-free troubleshooting support** while keeping users focused and free from distractions like phones or web browsing.

##

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Scope](#project-scope)
- [Acceptance Criteria](#acceptance-criteria)
- [Repository Structure](#repository-structure)
- [Credits](#credits)
- [License](#license)

##

## Installation

### Hardware Requirements
- Raspberry Pi 5  
- Raspberry Pi AI Camera (SC1174)  
- USB microphone  
- USB speaker  
- 18650 rechargeable battery system  
- Optional Raspberry Pi AI HAT+  
- 3D-printed housing  

### Software Requirements
- Python  
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
6. Power on and test interaction.

##

## Usage
1. Turn on the Pocket Professor.  
2. Ask an **IT-related** question using your voice.  
3. Show relevant workspace items to the camera if needed.  
4. Receive a **spoken hint** to guide troubleshooting.  
5. Place the device face-down to trigger an orientation reaction.

##

## Features
- Reliable **power management** with safe shutdown  
- **Live camera perception** of workspace  
- **Hands-free voice interaction**  
- **Hint-based AI responses** (no full answers)  
- **3D-printed professor housing** that stands upright  
- **Orientation detection** with audible reaction  
- Fully **local, open-source AI processing**

##

## Project Scope

### In Scope
- Physical 8–10 inch professor model  
- Raspberry Pi with open-source AI  
- Camera-based visual understanding  
- Voice input and output  
- Orientation detection and reactions  
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
- Stands upright independently  
- Responds to spoken IT questions  
- Provides relevant hints within **5 seconds**  
- Detects face-down placement and reacts audibly  
- Maintains consistent and appropriate responses  
