import certifi
import ssl

ssl.create_default_context(cafile=certifi.where())
print("Certificates updated successfully!")