import jwt
import ssl
import os
import base64
import json
import requests
from jwt import PyJWKClient
#from src.config.config import Config
from src.config.config import Config
from src.logging.app_logger import AppLogger
from werkzeug.wrappers import Request, Response, ResponseStream
from dotenv import dotenv_values


# DOCUMENTATION on middleware for Flask: https://medium.com/swlh/creating-middlewares-with-python-flask-166bd03f2fd4
class Authentication(object):

    config = dotenv_values(os.path.join("config", ".env"))

    auth_config = {
        "auth_cookie_name": config.get("auth_cookie_name"),
        "issuer": config.get("issuer"),
        "jwks_uri": config.get("jwks_uri"),
        "algorithm": config.get("algorithm"),
        "audience": config.get("audience")
    }

    def __init__(self, app) -> None:
        self.logger = AppLogger.get_logger()
        self.app = app
        self.logger.info(str(Authentication.auth_config))

    def __call__(self, environ, start_response):
        #self.logger.info("------------------------ Authentication -----------------------------")
        # for key,value in environ.items():
        #     self.logger.info(key + " -> " + str(value))

        request = Request(environ, shallow=False)

        # # ---- skip static assets early ----
        path = request.path
        if (
            path.startswith("/static/")
            or path.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".ico"))
        ):
            #self.logger.info(f"Skipping auth for static resource: {path}")
            #environ["status_code"] = 200
            return self.app(environ, start_response)
        # # ---- end static skip ----

        # skip all authentication logic for OPTIONS
        if request.method == "OPTIONS":
            #self.logger.info("request.method is " + str(request.method))
            #environ["status_code"] = 200
            return self.app(environ, start_response)

        # https://flask.palletsprojects.com/en/3.0.x/api/#flask.Request.path
        #self.logger.info("request.base_url " + str(request.base_url))
        #self.logger.info("request.path " + str(request.path))
        #self.logger.info("request.headers " + str(request.headers))
        #self.logger.info(str(request.headers.get("Cookie")))
        #self.logger.info("request.authorization " + str(request.authorization))
        #self.logger.info("request.host_url " + request.host_url)
        #self.logger.info("request.cookies -> " + str(request.cookies))
        #self.logger.info("request.args -> " + str(request.args))
        #self.logger.info("request.application -> " + str(request.application))
        #self.logger.info("request.data -> " + str(request.data))
        #self.logger.info("request.full_path -> " + str(request.full_path))

        # for key,value in environ.items():
        #     if key == "HTTP_COOKIE" or key == "HTTP_HOST" or key == "REQUEST_URI":
        #         self.logger.info(key + " -> " + str(value))
        
        #self.logger.info(f"Request path: {request.path}, full_path: {request.full_path}")
        #self.logger.info(f"Cookie header: {request.headers.get('Cookie')}")

        try:
            data = request.headers.get("Cookie") if request.headers.get("Cookie") is not None \
                    else (environ["HTTP_COOKIE"] if "HTTP_COOKIE" in environ else None)
            
            #self.logger.info("data: " + str(data))
            ###
            ### begin 1-time debug
            resp = requests.get(Authentication.auth_config["jwks_uri"], timeout=5)
            self.logger.info(f"JWKS status: {resp.status_code}")
            if resp.status_code == 200:
                jwks = resp.json()
                general_key = next((k for k in jwks['keys'] if k['kid']=='general'), None)
                if general_key:
                    self.logger.info(f"JWKS has 'general' key: alg={general_key['alg']}")
                else:
                    self.logger.info("❌ NO 'general' key in JWKS!")
            ### end 1-time debug

            JWT = self.extractJWT(data) if data is not None else None
            self.debugger(JWT)
            #if JWT is not None:
            #   self.logger.info("JWT is " + JWT)

            if JWT is None: # return 401
                self.logger.info("No JWT ... returning 401")
                environ["status_code"] = 401
                return self.app(environ, start_response)

            # https://pyjwt.readthedocs.io/en/stable/usage.html#retrieve-rsa-signing-keys-from-a-jwks-endpoint
            signing_key = self.extract_signing_key(JWT) #from https://go-dev.fuqua.duke.edu/auth/jwks

            # claims = jwt.decode(
            #     JWT,
            #     key=signing_key.key,
            #     algorithms=Authentication.auth_config["algorithm"], # RS256
            #     options={"verify_exp": True, "verify_iat": False}, # expiration time, issued at time
            #     audience=Authentication.auth_config["audience"], #"FuquaWorld",
            #     issuer=Authentication.auth_config["issuer"] # https://go-dev.fuqua.duke.edu/auth
            # )
            self.logger.info("=== STARTING JWT DECODE ===")
            self.logger.info(f"Using alg: {Authentication.auth_config['algorithm']}")

            claims = jwt.decode(
                JWT,
                key=signing_key.key,  # Use .key property
                algorithms=[Authentication.auth_config["algorithm"]],  # LIST format!
                options={"verify_exp": True, "verify_iat": False},
                audience=Authentication.auth_config["audience"],
                issuer=Authentication.auth_config["issuer"]
            )
            self.logger.info("DECODE SUCCESSFUL")

            if claims is None: # return 401
                self.logger.info("No claims taken from JWT ... returning 401")
                environ["status_code"] = 401
                return self.app(environ, start_response)

            #self.logger.info(str(claims))

            environ["status_code"] = 200
            environ["claims"] = claims
            #self.logger.info(str(environ["claims"]))

            return self.app(environ, start_response)

        except Exception as err:
            self.logger.error(str(err))
            if "Signature has expired".upper() in (str(err)).upper():
                environ["status_code"] = 401
                return self.app(environ, start_response)
            
            if "Signature verification failed".upper() in (str(err)).upper():
                environ["status_code"] = 401
                return self.app(environ, start_response)

            environ["status_code"] = 500
            environ["err"] = str(err)
            return self.app(environ, start_response)

    def extractJWT(self, cookieString:str) -> str:
        self.logger.info(f"Auth config: issuer={Authentication.auth_config['issuer']}, audience={Authentication.auth_config['audience']}, algorithm={Authentication.auth_config['algorithm']}, jwks_uri={Authentication.auth_config['jwks_uri']}")

        try:
            if cookieString is None or len(cookieString) == 0:
                self.logger.info("NO COOKIE STRING")
                return None

            cookies = cookieString.split(";")

            filtered = list(filter(lambda x: Authentication.auth_config["auth_cookie_name"] in x, cookies))
            if len(filtered) == 0:
                return None

            JWT = filtered[0].strip().replace(Authentication.auth_config["auth_cookie_name"]+"=", "")
            return JWT
        except Exception as err:
            self.logger.error(str(err))
            self.logger.error(str(err.__dict__))
            return None

    # def extract_signing_key(self, JWT:str):
    #     jwks_client = PyJWKClient(Authentication.auth_config["jwks_uri"] )
    #     signing_key = jwks_client.get_signing_key_from_jwt(JWT)
    #     return signing_key

    def extract_signing_key(self, JWT:str):
        self.logger.info(f"Fetching from JWKS: {Authentication.auth_config['jwks_uri']}")
        jwks_client = PyJWKClient(Authentication.auth_config["jwks_uri"])
        signing_key = jwks_client.get_signing_key_from_jwt(JWT)
        
        self.logger.info(f"KEY kid: {getattr(signing_key, 'key_id', 'MISSING')}")
        self.logger.info(f"KEY alg: {getattr(signing_key, 'algorithm', 'MISSING')}")
        self.logger.info(f"KEY matches header? {getattr(signing_key, 'key_id', 'MISSING') == 'general'}")
        return signing_key


    def debugger(self, JWT):
        self.logger.info(f"Extracted JWT (first 50 chars): {JWT[:50]}...")
        if JWT:
            # Decode header (base64) without verification to inspect
            try:
                header_b64 = JWT.split('.')[0] + '=' * (4 - len(JWT.split('.')[0]) % 4)
                header = json.loads(base64.b64decode(header_b64).decode())
                self.logger.info(f"JWT Header: kid={header.get('kid')}, alg={header.get('alg')}, typ={header.get('typ')}")
            except Exception as e:
                self.logger.error(f"Failed to decode JWT header: {e}")
