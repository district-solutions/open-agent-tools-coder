import os
from oats.log import cl

log = cl('ow.ga')


def get_auth_env(
    url: str | None = None,
    email: str | None = None,
    password: str | None = None,
    env_name: str | None = None,
    auth_type: str | None = None,
    target_oweb: str | None = None,
    verbose: bool = False,
):
    """
    get auth credentials returned as a tuple (url, email, password) for an env and auth type

    :param url: fqdn or https://fqdn
    :param email: email to use
    :param password: password to use
    :param env_name: name of environment
    :param auth_type: type of auth
    :param target_oweb: oweb alias
    :param verbose: log more if enabled
    """
    base_url = url
    use_auth = target_oweb
    if use_auth is None:
        use_auth = auth_type
    if verbose and env_name is None:
        log.info(f'auth: {use_auth}')
    if base_url is None:
        base_address = os.getenv("CODER_CHAT_URL", None)
        if base_address is None:
            err = f'### Sorry!! Please set these environment variables for open-webui get_auth:\n```\nexport CODER_CHAT_URL=http://0.0.0.0:PORT\nexport CODER_CHAT_EMAIL=user@email.com\nexport CODER_CHAT_PASSWORD=PW\n```\n'
            raise Exception(err)
        if '192.168.' not in base_address and '0.0.0.0' not in base_address and 'localhost' not in base_address:
            if 'https://' not in base_address and 'http://' not in base_address:
                base_url = f"https://{base_address}"
        else:
            if base_address.count('.') == 4 and 'http://' not in base_address:
                base_url = f"http://{base_address}"
        if "https://" in base_address or 'http://' in base_address:
            base_url = base_address
    if base_url is not None:
        base_url = base_url.replace('http://http://', 'http://').replace('https://http://', 'http://').replace('https://https://', 'https://')
    email = os.getenv('CODER_CHAT_EMAIL', email)
    password = os.getenv('CODER_CHAT_PASSWORD', password)
    if email is None:
        err = f'### Sorry!! Please set these environment variables - missing email for get_auth:\n```\nexport CODER_CHAT_URL=http://0.0.0.0:PORT\nexport CODER_CHAT_EMAIL=user@email.com\nexport CODER_CHAT_PASSWORD=PW\n```\n'
        raise Exception(err)
    if password is None:
        err = f'### Sorry!! Please set these environment variables - missing password for get_auth:\n```\nexport CODER_CHAT_URL=http://0.0.0.0:PORT\nexport CODER_CHAT_EMAIL=user@email.com\nexport CODER_CHAT_PASSWORD=PW\n```\n'
        raise Exception(err)
    return base_url, email, password
