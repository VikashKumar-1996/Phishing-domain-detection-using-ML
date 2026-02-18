import re
import socket
import ssl
import requests

from urllib.parse import urlparse


# helper functions

def has_ip(url):
    match = re.search(r"\d+\.\d+\.\d+\.\d+", url)
    return -1 if match else 1


def url_length(url):
    if len(url) < 54:
        return 1
    elif len(url) <= 75:
        return 0
    else:
        return -1


def has_at_symbol(url):
    return -1 if "@" in url else 1


def double_slash_redirect(url):
    return -1 if "//" in urlparse(url).path else 1


def prefix_suffix(domain):
    return -1 if "-" in domain else 1


def subdomain_count(domain):

    dots = domain.count(".")

    if dots == 1:
        return 1
    elif dots == 2:
        return 0
    else:
        return -1


def https_token(domain):
    return -1 if "https" in domain else 1


def request_url(url):

    try:
        r = requests.get(url, timeout=3)
        return 1 if r.status_code == 200 else -1
    except:
        return -1


def check_port(domain):

    try:
        socket.gethostbyname(domain)
        return 1
    except:
        return -1


def ssl_state(domain):

    try:
        ssl.create_default_context().wrap_socket(
            socket.socket(),
            server_hostname=domain,
        )
        return 1
    except:
        return -1


# main extractor

def extract_features(url):

    parsed = urlparse(url)

    domain = parsed.netloc

    features = []

    # 1
    features.append(has_ip(url))

    # 2
    features.append(url_length(url))

    # 3
    features.append(has_at_symbol(url))

    # 4
    features.append(double_slash_redirect(url))

    # 5
    features.append(prefix_suffix(domain))

    # 6
    features.append(subdomain_count(domain))

    # 7
    features.append(ssl_state(domain))

    # 8
    features.append(https_token(domain))

    # 9
    features.append(request_url(url))

    # 10
    features.append(check_port(domain))



    while len(features) < 30:
        features.append(1)

    return features
