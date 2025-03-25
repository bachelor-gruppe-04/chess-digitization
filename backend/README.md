# Backend

Contains multiple elements of the application in its whole. Elements such as Camera logic, machine learning algorithms, and WebSocket hosting.

## Requirements

To install the necessary requirements for the backend to run, please navigate to the ``backend`` folder from the root directory using :

    cd backend

Then, when the directory points to the backend folder, please run the following line in the terminal of the directory:

    pip install -r requirements.txt


## How to run

To start the backend, *assuming the current directory includes the backend folder,* one has to run these following lines:

### WebSockets

for starting the WebSocket and cameras:

    uvicorn api:app --reload

### Machine Learning Algorithm

for starting the machine learning algorithm:

    python logic/detection/run_video.py