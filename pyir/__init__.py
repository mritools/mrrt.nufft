# see https://docs.python.org/3/library/pkgutil.html
# from pkgutil import extend_path
# __path__ = extend_path(__path__, __name__)

# switched to setuptools solution based on pkg_resources
__import__("pkg_resources").declare_namespace(__name__)
