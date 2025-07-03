import os
import sys

# Set the path to your application directory (your repo folder)
sys.path.insert(0, os.path.join('/home/wwwmayam/repositories/aceh-gold-forecast'))

# Activate your virtual environment
# Make sure this path is correct relative to your app directory
INTERP = os.path.join('/home/wwwmayam/repositories/aceh-gold-forecast/virtualenv/bin/python')
if sys.executable != INTERP:
    os.execl(INTERP, INTERP, *sys.argv)

# Import your FastAPI app instance
# Assuming your main FastAPI app is defined in 'main.py' within your 'api' folder
from api.main import app as application # Adjust 'api.main' and 'app' as per your FastAPI app structure

# For Gunicorn/Uvicorn usage, this part is just a placeholder,
# Passenger will run your app using your server command.
# You might not even need the 'application' variable if Passenger is configured with a custom command.
# Often, you'd use a command in your 'run.sh' and configure Passenger to use that.