# Backend

Contains multiple elements of the application in its whole. Elements such as camera logic, machine learning algorithms, and WebSocket hosting.

## Requirements

To install the necessary requirements for the backend to run, please navigate to the ``backend`` folder from the root directory using :

    cd backend

Then, when the directory points to the backend folder, please run the following line in the terminal of the directory:

    pip install -r requirements.txt


## How to run

To start the backend, run these following lines:

### WebSockets

For starting the WebSocket and cameras, navigate to the correct directory:

    cd logic/api

Then, when the directory includes the correct path, run the following line to start the process:

    uvicorn api:app --reload

### Machine Learning Algorithm

For starting the algorithm, make sure the directory point to the ``backend`` folder, then run the following line:

    python logic/detection/run_video.py