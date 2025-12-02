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

# DOCUMENTATION on PyJWT: https://pypi.org/project/PyJWT/
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
        #self.logger.info(str(Authentication.auth_config))

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
            environ["status_code"] = 200
            return self.app(environ, start_response)
        # # ---- end static skip ----

        # skip all authentication logic for OPTIONS
        if request.method == "OPTIONS":
            #self.logger.info("request.method is " + str(request.method))
            environ["status_code"] = 200
            return self.app(environ, start_response)
        
        # begin debug
        response = requests.get(Authentication.auth_config["jwks_uri"])
        response.raise_for_status()  # Raise an exception for HTTP errors

        jwks = response.json()  # Parse the JWKS as JSON
        print(json.dumps(jwks, indent=2))  # Pretty-print the keys

        # Optional: inspect key IDs ("kid")
        if "keys" in jwks:
            key_ids = [key.get("kid") for key in jwks["keys"]]
            print("Available key IDs:", key_ids)
        else:
            print("No 'keys' field found in JWKS response!")
        # end debug

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

            JWT = self.extractJWT(data) if data is not None else None
            #if JWT is not None:
            #   self.logger.info("JWT is " + JWT)

            if JWT is None: # return 401
                self.logger.info(str(request.full_path) + " " + "No JWT ... returning 401")
                environ["status_code"] = 401
                return self.app(environ, start_response)
            
            # begin debug
            header = jwt.get_unverified_header(JWT)
            self.logger.info("JWT header kid: " + str(header.get("kid")))

            claims_unverified = jwt.decode(JWT, options={"verify_signature": False})
            self.logger.info(f"Unverified claims: {claims_unverified}")
            # end debug

            # https://pyjwt.readthedocs.io/en/stable/usage.html#retrieve-rsa-signing-keys-from-a-jwks-endpoint
            signing_key = self.extract_signing_key(JWT) 

            # claims = jwt.decode(
            #     JWT,
            #     key=signing_key.key,
            #     algorithms=Authentication.auth_config["algorithm"], # RS256
            #     options={"verify_exp": True, "verify_iat": False}, # expiration time, issued at time
            #     audience=Authentication.auth_config["audience"], #"FuquaWorld",
            #     issuer=Authentication.auth_config["issuer"] # https://go-dev.fuqua.duke.edu/auth
            # )
            #self.logger.info("=== STARTING JWT DECODE ===")
            #self.logger.info(f"Using alg: {Authentication.auth_config['algorithm']}")

            claims = jwt.decode(
                JWT,
                key=signing_key.key,  # Use .key property
                algorithms=[Authentication.auth_config["algorithm"]],  # LIST format!
                # options={"verify_exp": True, "verify_iat": False},
                options={"verify_exp": True, "verify_iat": False, "verify_aud": False, "verify_signature": False},
                audience=Authentication.auth_config["audience"],
                issuer=Authentication.auth_config["issuer"],
            )
            #self.logger.info(str(request.full_path) + " DECODE SUCCESSFUL")

            if claims is None: # return 401
                self.logger.info(str(request.full_path) + " No claims taken from JWT ... returning 401")
                environ["status_code"] = 401
                return self.app(environ, start_response)

            #self.logger.info(str(claims))

            environ["status_code"] = 200
            environ["claims"] = claims
            #self.logger.info(str(environ["claims"]))

            return self.app(environ, start_response)

        except Exception as err:
            self.logger.error(str(request.full_path) + " " + str(err))
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

    def extract_signing_key(self, JWT:str):
        jwks_client = PyJWKClient(Authentication.auth_config["jwks_uri"] )
        signing_key = jwks_client.get_signing_key_from_jwt(JWT)
        return signing_key
