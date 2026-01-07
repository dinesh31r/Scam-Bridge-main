# gen.py
import importlib.metadata  # built-in on Python 3.10+

# --- optional defensive patch for older 3.9 environments ---
if not hasattr(importlib.metadata, "packages_distributions"):
    try:
        import importlib_metadata  # only exists if back-port installed
        importlib.metadata.packages_distributions = importlib_metadata.packages_distributions
    except ImportError:
        pass
# ------------------------------------------------------------

import google.generativeai as genai

genai.configure(api_key="Insert your api key here")

model = genai.GenerativeModel("models/gemini-2.5-flash")

response = model.generate_content(
    "Explain the concept of network anomaly detection in one paragraph."
)
print(response.text)








