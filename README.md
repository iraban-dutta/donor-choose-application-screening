## End to End ML Project: DonorsChoose Application Screening



------------------------------------------------------------------------------------------------
### Steps to run the web application on your local machine:
1. Clone the git repository
```bash
git clone <repository-url>
```
2. Descend into the cloned directory and create a virtual environment
```bash
cd <repository-name>

python -m venv venv
```
3. Activate the virtual environment
- macOS:
```bash
source venv/bin/activate
```
- Windows:
```bash
venv\Scripts\activate
```
4. Install the required libraries
```bash
pip install -r requirements.txt
```
5. Start the Flask server
```bash
python app.py
```
6. Access the web application locally at http://127.0.0.1:5000/
------------------------------------------------------------------------------------------------


### Add project path to $PYTHONPATH environment variable:
- In terminal navigate to the project root path (eg: '/path/to/project') and follow the below steps
- Execute the following statement: export PYTHONPATH=$PYTHONPATH:/path/to/project

### Install requirements:
- Install packages: pip install -r requirements.txt
