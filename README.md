# Financial Analyzer
### Updated December 15, 2025

The app can run in two modes, localhost/single-user, or non-localhost/multi-user.

#### For localhost: 

Start the React frontend with ```npm start```, then start the Python/Flask backend with ```python app.py```.
In browser, navigate to ```localhost:3000```.

The backend routes are served by ```Flask```.

#### For non-localhost:

The app is served at ```https://go-dev.fuqua.duke.edu/financial_analyzer``` (test system) and ```https://go.fuqua.duke.edu/financial_analyzer``` (live system).

The backend API routes are served by ```gunicorn```, which is a Python WSGI HTTP server for Linux systems.  It provides support for multiple worker processes to handle concurrent user requests.

When running under ```gunicorn```, the application entry point is ```wsgi.py``` instead of ```app.py```.

#### Backend project changes required to support non-localhost access:

1. ```app.py``` no longer contains the API endpoint definitions.  Its new purpose is to provide backend initialization depending on in which mode the app is running (localhost/Flask vs. non-localhost/gunicorn).  

There is a new folder, ```src/routes```.  This is the location for all of the API endpoint code that was previously in ```app.py```.

The ```src/routes/api_routes.py``` code contains the endpoints for launching the app (requesting ```index.html``` and being redirected to ```/mv```).

The ```src/routes/backtest.py``` code contains all endpoints related to backtest.  Similarly for lifecycle, matrix, and regression.

All of the API routes have been adapted to use the Flask blueprint framework.  The reason for this choice was that with blueprints (a blueprint equals a collection of 1 or more API endpoints), it is possible to designate a static folder that is visible only to the specific blueprint.  For example, the definition for the lifecycle blueprint:

```
        self.blueprint = Blueprint(
            "LifeCycle",
            __name__,
            url_prefix=f"{self.APP_PREFIX}/lifecycle",
            static_url_path="/static",     # served at /lifecycle/static
            static_folder=static_dir,
        )
```

File names for file uploads and downloads now embed a userID within the file name, to ensure each user works with their own set of upload/download files.  These files are uploaded/downloaded based on the URL path, here, ```/lifecycle/static```.   Matrix-related file uploads/downsload are served at ```/matrix/static```, and so on.

2. Authentication middleware is in place to support user authentication when running the app in non-localhost mode.  See ```src/middleware```.  This is code to intercept every request that comes to the backend and ensure the user sending the request has been authenticated.  This code does not run in localhost mode.

3. Configuration and logging support have been added (see ```src/config```, ```src/logging```).

#### Frontend project changes required to support non-localhost access:

The required changes were very minimal compared to the backend code base.  All changes made were to support user authentication.

The ```index.js``` file added this import: ```import '@fsb/fw-auth/dist/fw-auth.js'```.  This is web component code that identifies the user's Duke credentials (netid, userid, etc.).

The ```App.jsx``` file now tests to see if the user is running under localhost.  If the answer is no, the ```<fw-auth>``` web component is returned with the page load.   What this means: If running under localhost, no user authentication happens, otherwise, when the user first attempts to hit the home page in non-localhost mode, the user will be redirected to a FuquaWorld login screen.

# Server deployment 

The app is dockerized (see ```fa.Dockerfile```).  It runs in a Docker container.

Gitlab's continuous integration/continuous deployment (CI/CD) capability is used for deployment (see ```.gitlab-ci.yml```).  It is currently active for the ```dev``` branch and the ```main``` branch.  It is not relevant for the ```master``` branch, at this time.

What this means: pushing changes into the ```main``` branch (for example):

```
git add .
git commit -m "my commit message"
git push
```

automatically triggers the build job defined in ```.gitlab-ci.yml```.  This job will stop the running docker container instance on the server, build a new docker image based on the main branch content, push it onto the server and start the new container running.

The build job is configured (at the moment) to only push out to the test server (serving over ```https://go-dev.fuqua.duke.edu/financial_analyzer```).

To push to the live server (```https://go.fuqua.duke.edu/financial_analyzer```), an additional (as yet not created) deployment step will be required.  These instructions will be updated when the live deployment resources are all created.

Important!  If doing local develpment and it is not desired to re-deploy the app on the server, add the message ```[ci skip]```, like this:

```
git add .
git commit -m [ci skip]
git push
```

This will commit changes, but not re-deploy the app.


VOO,VXUS,AVUV,AVDV,AVEM

# Backend

## install Python 3.13

## Run in terminal
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

## Put necessary csv file under the backend folder
```
'F-F_Research_Data_Factors.csv'
'stocks_mf_ETF_data_final.csv'
'F-F_Momentum_Factor.csv'
'F-F_Research_Data_5_Factors_2x3.csv'
```

## start backend
```bash
python app.py
```

## troubleshooting (optional)

### problem #1 cannot recognize python vs python3

<b>solution</b>

setup alias (for *nix)
```
alias python=python3
```

### problem #2 environment 


```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

```

# Front End

## Create .env file under frontend directory

```
REACT_APP_API_BASE_URL=http://localhost:5001
```

## install dependency

```
npm install
```

## start frontend

```
npm start
```

