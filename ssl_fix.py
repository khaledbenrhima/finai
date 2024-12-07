#Update root SSL certificate command to run on terminal:
# /Applications/Python\ 3.12/Install\ Certificates.command

import ssl
import certifi
import urllib.request

def check_ssl():
    print(f"Python's default SSL context uses: {ssl.get_default_verify_paths().cafile}")
    print(f"Certifi's SSL context uses: {certifi.where()}")

    try:
        urllib.request.urlopen('https://github.com')
        print("SSL is working correctly.")
    except urllib.error.URLError as e:
        print(f"SSL is not working. Error: {e}")

    try:
        context = ssl.create_default_context(cafile=certifi.where())
        urllib.request.urlopen('https://github.com', context=context)
        print("SSL is working with certifi's certificates.")
    except urllib.error.URLError as e:
        print(f"SSL is not working even with certifi. Error: {e}")

if __name__ == "__main__":
    check_ssl()