import os
import requests
import traceback
from oats.log import cl

log = cl('oweb.lg')


def login_to_openwebui(
    email: str,
    password: str,
    base_url: str = None,
    verbose: bool = False,
):
    """
    login to the openweb ui api and get the user's api token

    Example

    ```python
    base_address = os.getenv("CODER_CHAT_URL", "api3.obuild.ai")
    email = os.getenv("CODER_CHAT_EMAIL", "email@email.com")
    password = os.getenv("CODER_CHAT_PASSWORD", "123321")
    login_dict = login_to_openwebui(email, password)
    print(login_dict.get("token", "no-token-found")
    ```

    :param email: user email
    :param password: user password
    :param base_url: url for openweb ui
    :param verbose: log more if enabled
    :return: dictionary from openweb ui login
    """
    base_address = os.getenv("CODER_CHAT_URL", base_url)
    if base_address is None:
        err = f'### Sorry!! Please set these environment variables for open-webui login:\n```\nexport CODER_CHAT_URL=http://0.0.0.0:PORT\nexport CODER_CHAT_EMAIL=user@email.com\nexport CODER_CHAT_PASSWORD=PW\n```\n'
        raise Exception(err)
    if '192.168.' not in base_address and '0.0.0.0' not in base_address and 'localhost' not in base_address:
        if 'https://' not in base_address:
            base_url = f"https://{base_address}"
        else:
            if base_address.count('.') == 4 and 'http://' not in base_address:
                base_url = f"http://{base_address}"
    if "https://" in base_address or 'http://' in base_address:
        base_url = base_address
    base_url = base_url.replace('http://http://', 'http://').replace('https://http://', 'http://').replace('https://https://', 'https://')
    # log.critical(f'### Debug Oweb Auth\n\nbase_url:\n```\n{base_url}\n```')
    email = os.getenv('CODER_CHAT_EMAIL', email)
    password = os.getenv('CODER_CHAT_PASSWORD', password)
    if email is None:
        err = f'### Sorry!! Please set these environment variables - missing email for open-webui login::\n```\nexport CODER_CHAT_URL=http://0.0.0.0:PORT\nexport CODER_CHAT_EMAIL=user@email.com\nexport CODER_CHAT_PASSWORD=PW\n```\n'
        raise Exception(err)
    if password is None:
        err = f'### Sorry!! Please set these environment variables - missing password for open-webui login::\n```\nexport CODER_CHAT_URL=http://0.0.0.0:PORT\nexport CODER_CHAT_EMAIL=user@email.com\nexport CODER_CHAT_PASSWORD=PW\n```\n'
        raise Exception(err)
    auth_data = {"email": email, "password": password}
    retry_session = requests.Session()
    retry_session.headers.update({"Content-Type": "application/json"})
    try:
        if verbose:
            log.info(f"login {email} to {base_url}/api/v1/auths/signin")
        # log.info(f"#### OWEB login url: {base_url} with email: {email} pw: {password}")
        response = retry_session.post(f"{base_url}/api/v1/auths/signin", json=auth_data, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except Exception as e:
        if '400 client error: bad request' in str(e).lower():
            # log.error(f"### Sorry!! failed BAD_REQUEST_LOGIN to: {base_url} with email: {email} pw: {password} with error:\n```\n{traceback.format_exc()}\n```\n\nCheck the Env Variables\n\n")
            log.error(f"### Sorry!! failed BAD_REQUEST_LOGIN to: {base_url} with email: {email} with error:\n```\n{traceback.format_exc()}\n```\n\nCheck the Env Variables: CODER_CHAT_URL, CODER_CHAT_EMAIL, CODER_CHAT_PASSWORD\n\n")
        else:
            log.error(f"### Sorry!! failed to login to: {base_url} with email: {email} with error:\n```\n{traceback.format_exc()}\n```\n\nCheck the Env Variables: CODER_CHAT_URL, CODER_CHAT_EMAIL, CODER_CHAT_PASSWORD")
        return None
