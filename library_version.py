import importlib
import pkg_resources

# List of packages from requirements.txt
packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
    "emoji",
    "bs4",
    "textblob",
    "nltk",
    "wordcloud",
    "distance",
    # "python-Levenshtein",
    "Levenshtein",
    "fuzzywuzzy",
    "tqdm",
    # "scikit-learn",
    "sklearn",
    "statsmodels",
    "category_encoders",
    "imblearn",
    "xgboost",
    "gensim",
    "flask",
    "gunicorn",
]

# Function to print package version
def print_version(package):
    try:
        module = importlib.import_module(package)
        if package == "sklearn":
            version = module.__version__
        else:
            version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except ImportError:
        print(f"{package}: Not installed")
    except pkg_resources.DistributionNotFound:
        print(f"{package}: Version information not found")

# Iterate through the packages and print the version
for package in packages:
    print_version(package)




# # Add project path to $PYTHONPATH environment variable:
# - In terminal navigate to the project root path (eg: '/path/to/project') and follow the below steps
# - Execute the following statement: export PYTHONPATH=$PYTHONPATH:/path/to/project


# Demo Video: https://www.loom.com/share/a28030c9d10b4716a7eab01729f41e9b?sid=b318b980-266a-414e-bb21-ffbe3d8348d3