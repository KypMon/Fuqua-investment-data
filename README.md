hello world

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

