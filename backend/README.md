# Backend

Contains multiple elements of the application in its whole. Elements such as camera logic, machine learning algorithms, and WebSocket hosting.

## Requirements

To install the necessary requirements for the backend to run, please navigate to the ``backend`` folder from the root directory using :

    cd backend

Then, when the directory points to the backend folder, please run the following line in the terminal of the directory:

    pip install -r requirements.txt


## How to run

To start the backend, run these following lines:

    cd backend

Then, when the directory includes the correct path, run the following line to start the process:

    uvicorn main:app --reload

## How to run tests

to run the tests for the backend logic, run these following lines:

    cd backend

Then, when the directory is in the correct path, run the following line to start the test process:

    python -m unittest logic.api.entity.test_board