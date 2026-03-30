# Back_end

This is the part of the project where all the logic related to collecting data to build up the analysis, and the way I export all the interfaces to get access to the Machine Learning Models in order to predict results that will be necesary for a client side to verify a transation as possible fraudulent or not.

<p>&nbsp;</p>

**Installation:**

Create a virtual environment inside the back_end folder:

> python3 -m venv .venv

All the necesary libraries needed for this Python project are located in the `requirements.txt` file, and the way to start installation of those libraries is running this command line:

> pip install -r requirements.txt

<p>&nbsp;</p>

**Running Back_end:**

This Python project is based on FastAPI library to create a self-service server provider, that means that we can create simple REST API server running this command after locating into the folder `back_end`:

> uvicorn back_end.src.app.main:app --reload

<p>&nbsp;</p>

**Running Sawgger Interface:**

FastAPI already has an internal implementation of Swagger and there is no need to install external libraries, the way we can access the swagger interfaz is through this URL after the project is up and running:

> http://127.0.0.1:8000/docs


